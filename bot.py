"""Translate new posts from Telegram channels into English and repost them.

Runs once and exits: designed to be started by a scheduler (GitHub Actions cron).
Progress per channel is kept in state.enc (encrypted with STATE_KEY), so every
run picks up where the previous one stopped.

Privacy: the repository and the Actions logs may be public. The list of channels
and the target chat come only from secrets (TG_CHANNELS, TG_TARGET), the state
file is encrypted, and logs never print channel names, links or post text.
"""

import asyncio
import base64
import hashlib
import html
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml
from cryptography.fernet import Fernet, InvalidToken
from telethon import TelegramClient
from telethon.sessions import StringSession

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
STATE_PATH = ROOT / "state.enc"

TEXT_LIMIT = 4096
CAPTION_LIMIT = 1024
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

DRY_RUN = os.environ.get("DRY_RUN") == "1"


# ---------------------------------------------------------------- config/state

def parse_list(value):
    return [x.strip() for x in re.split(r"[,\n]", value or "") if x.strip()]


def load_config(path=CONFIG_PATH, env=os.environ):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    # Private settings come from secrets and override anything in the public file.
    if env.get("TG_CHANNELS"):
        cfg["channels"] = parse_list(env["TG_CHANNELS"])
    if env.get("TG_TARGET"):
        cfg["target"] = env["TG_TARGET"].strip()
    cfg.setdefault("channels", [])
    cfg.setdefault("target", "me")
    cfg.setdefault("level", "B2")
    cfg.setdefault("vocab_words", 5)
    cfg.setdefault("show_original", True)
    cfg.setdefault("max_posts_per_run", 15)
    cfg.setdefault("initial_posts", 3)
    models = cfg.get("models") or [cfg.get("model") or "gemini-flash-latest"]
    cfg["models"] = [models] if isinstance(models, str) else list(models)
    cfg.setdefault("seconds_between_requests", 6)
    cfg.setdefault("max_run_seconds", 600)  # stop taking new posts after this; job limit is 15 min
    cfg["channels"] = [str(c).strip().lstrip("@") for c in cfg["channels"] if str(c).strip()]
    return cfg


def _fernet():
    key = os.environ.get("STATE_KEY", "").strip()
    if not key:
        raise KeyError("STATE_KEY")
    # Accept any passphrase: derive a proper Fernet key from it.
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(key.encode()).digest()))


def load_state(path=None):
    p = Path(path or STATE_PATH)
    if not p.exists() or not p.read_bytes().strip():
        return {}
    try:
        return json.loads(_fernet().decrypt(p.read_bytes().strip()))
    except InvalidToken:
        print("state.enc cannot be decrypted (STATE_KEY changed?), starting fresh")
        return {}


def save_state(state, path=None):
    if DRY_RUN:
        return
    Path(path or STATE_PATH).write_bytes(_fernet().encrypt(json.dumps(state).encode()) + b"\n")


# ---------------------------------------------------------------- translation

class RateLimited(Exception):
    """Free-tier quota is exhausted: stop this run, continue on the next one."""


class TranslationFailed(Exception):
    pass


SYSTEM_PROMPT = """You translate Telegram news posts into English for a reader who is learning English (level {level}).

Rules:
- The source is usually Russian, sometimes Ossetian or a mix. Translate everything into natural, fluent English, the way a quality English-language news outlet would write it. Keep the meaning, tone and facts exactly; do not summarise, add or comment.
- The text is Telegram HTML. Keep every HTML tag (<b>, <i>, <u>, <s>, <a href="...">, <code>, <pre>, <blockquote>) in place around the translated words. Never change URLs. Keep emojis, hashtags and line breaks.
- Transliterate Russian names the standard English way (e.g. Набиуллина -> Nabiullina). Use established English names for organisations (ЦБ -> the Central Bank of Russia).
- Then pick {vocab_words} words or phrases from YOUR English translation that are most useful for a {level} learner (idioms, collocations, economic or political terms; skip trivial words). For each give "en" (as it appears in the translation) and "ru" (a short Russian equivalent). If the post is very short, give fewer or none.

Return JSON: {{"translation": "...", "vocabulary": [{{"en": "...", "ru": "..."}}]}}"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "translation": {"type": "STRING"},
        "vocabulary": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {"en": {"type": "STRING"}, "ru": {"type": "STRING"}},
                "required": ["en", "ru"],
            },
        },
    },
    "required": ["translation", "vocabulary"],
}


class Unavailable(Exception):
    """Gemini is overloaded or not answering: stop the run without marking
    posts as done, they are retried on the next run."""


class ProviderError(Exception):
    """Key, model or account problem: no post can be translated, so stop the run
    without marking posts as done (they are retried on the next run)."""


def _request_gemini(text, cfg, api_key, session, model):
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT.format(
            level=cfg["level"], vocab_words=cfg["vocab_words"])}]},
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {
            "temperature": 0.3,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }
    url = GEMINI_URL.format(model=model)
    r = session.post(url, json=body, headers={"x-goog-api-key": api_key}, timeout=60)
    extract = lambda d: "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
    return r, extract


def parse_result(raw):
    raw = raw.strip()
    if not raw.startswith("{"):  # some models wrap JSON in ```json fences or prose
        m = re.search(r"\{.*\}", raw, re.S)
        raw = m.group(0) if m else raw
    result = json.loads(raw)
    if not str(result.get("translation", "")).strip():
        raise ValueError("empty translation")
    vocab = result.get("vocabulary") or []
    result["vocabulary"] = [v for v in vocab if isinstance(v, dict)]
    return result


def translate(text, cfg, api_key, session=requests):
    """Try the configured models in turn: when one is overloaded (503) or times
    out, the next one is used right away."""
    models = cfg["models"]
    last_error, unavailable = None, 0
    for attempt in range(max(4, 2 * len(models))):
        model = models[attempt % len(models)]
        try:
            r, extract = _request_gemini(text, cfg, api_key, session, model)
        except requests.RequestException as e:
            unavailable += 1
            print(f"  {model}: network error ({type(e).__name__}), trying again")
            time.sleep(3)
            continue
        if r.status_code == 429:
            # Daily limit: give up until next run. Per-minute limit: wait and retry.
            if "PerDay" in r.text or "per day" in r.text.lower():
                raise RateLimited(r.text[:300])
            m = re.search(r'"retryDelay":\s*"(\d+)', r.text)
            wait = min(int(m.group(1)) + 1, 65) if m else 20
            print(f"  {model}: 429 rate limit, waiting {wait}s")
            time.sleep(wait)
            unavailable += 1
            continue
        if r.status_code >= 500:
            unavailable += 1
            print(f"  {model}: HTTP {r.status_code} (overloaded), trying again")
            time.sleep(3)
            continue
        if r.status_code in (401, 402, 403):
            raise ProviderError(f"HTTP {r.status_code}: {r.text[:400]}")
        if r.status_code != 200:  # 400/404: bad model name or request
            raise ProviderError(f"{model}: HTTP {r.status_code}: {r.text[:400]}")
        try:
            result = parse_result(extract(r.json()))
            result["model"] = model
            return result
        except (KeyError, IndexError, TypeError, ValueError) as e:
            last_error = f"bad response ({type(e).__name__}: {e})"[:300]
            continue
    if unavailable and last_error is None:
        raise Unavailable("all models overloaded or not answering")
    raise TranslationFailed(last_error or "unknown error")


# ---------------------------------------------------------------- formatting

def visible_len(html_text):
    """Approximate length Telegram counts (text without tags, entities decoded)."""
    return len(html.unescape(re.sub(r"<[^>]+>", "", html_text)))


def build_post(title, link, translation, vocabulary, original, show_original):
    header = f"<b>{html.escape(title)}</b>"
    parts = [header, translation.strip()]
    if vocabulary:
        lines = [f"• <b>{html.escape(v['en'])}</b> — {html.escape(v['ru'])}"
                 for v in vocabulary if v.get("en") and v.get("ru")]
        if lines:
            parts.append("📚 <b>Vocabulary</b>\n" + "\n".join(lines))
    footer = f'<a href="{html.escape(link, quote=True)}">Original</a>' if link else ""
    if footer:
        parts.append(footer)
    text = "\n\n".join(parts)
    if show_original and original.strip():
        with_original = text + "\n\n<blockquote expandable>" + original.strip() + "</blockquote>"
        if visible_len(with_original) <= TEXT_LIMIT:
            return with_original
    return text


def split_message(text, limit=TEXT_LIMIT):
    """Split on paragraph (then line) boundaries so each chunk fits into one message."""
    if visible_len(text) <= limit:
        return [text]
    chunks, current = [], ""
    for para in re.split(r"(\n\n)", text):
        candidate = current + para
        if visible_len(candidate) <= limit:
            current = candidate
            continue
        if current.strip():
            chunks.append(current.strip())
        current = para
        while visible_len(current) > limit:  # a single huge paragraph: cut by lines/chars
            cut = current.rfind("\n", 0, limit)
            cut = cut if cut > 0 else limit
            chunks.append(current[:cut].strip())
            current = current[cut:]
    if current.strip():
        chunks.append(current.strip())
    return chunks


def group_messages(messages):
    """Collapse albums (several messages with one grouped_id) into one post."""
    groups, index = [], {}
    for m in messages:
        gid = getattr(m, "grouped_id", None)
        if gid and gid in index:
            groups[index[gid]].append(m)
        else:
            if gid:
                index[gid] = len(groups)
            groups.append([m])
    return groups


# ---------------------------------------------------------------- telegram

def message_link(entity, msg_id):
    username = getattr(entity, "username", None)
    if username:
        return f"https://t.me/{username}/{msg_id}"
    return f"https://t.me/c/{entity.id}/{msg_id}"


async def resolve(client, ref, dialogs_cache):
    """Resolve a channel by @username, numeric id or exact title (from your dialogs)."""
    if ref in ("me", "self"):
        return "me"
    if not re.fullmatch(r"-?\d+", ref):
        try:
            return await client.get_entity(ref)
        except (ValueError, TypeError):
            pass
    if not dialogs_cache:
        async for d in client.iter_dialogs():
            dialogs_cache.append(d)
    for d in dialogs_cache:
        if str(d.id) == ref or str(d.entity.id) == ref or d.name == ref:
            return d.entity
    raise ValueError("chat not found (check the name/id and that you are subscribed)")


async def send_post(client, target, text, photos):
    if DRY_RUN:
        print("----- DRY RUN -----\n" + text + (f"\n[+{len(photos)} photo(s)]" if photos else "") + "\n")
        return
    if photos:
        try:
            if visible_len(text) <= CAPTION_LIMIT:
                await client.send_file(target, photos if len(photos) > 1 else photos[0],
                                       caption=text, parse_mode="html")
                return
            await client.send_file(target, photos if len(photos) > 1 else photos[0])
        except Exception as e:  # e.g. the source channel forbids saving content
            print(f"  could not resend media ({type(e).__name__}); sending text only")
    for chunk in split_message(text):
        await client.send_message(target, chunk, parse_mode="html", link_preview=False)


async def process_channel(client, ref, cfg, state, target, api_key, budget, dialogs_cache, label="channel"):
    entity = await resolve(client, ref, dialogs_cache)
    title = getattr(entity, "title", ref)
    last_id = state.get(ref)

    if last_id is None:
        msgs = [m async for m in client.iter_messages(entity, limit=cfg["initial_posts"] * 4)]
        msgs.reverse()
        groups = group_messages(msgs)[-cfg["initial_posts"]:]
    else:
        msgs = [m async for m in client.iter_messages(entity, min_id=last_id, reverse=True, limit=200)]
        groups = group_messages(msgs)

    print(f"{label}: {len(groups)} new post(s)")  # no names: Actions logs may be public
    if not groups and last_id is None and msgs:
        state[ref] = msgs[-1].id

    sent = 0
    for group in groups:
        if sent >= budget:
            break
        if time.monotonic() > cfg.get("_deadline", float("inf")):
            print("  run time limit reached, the rest goes next run")
            break
        max_id = max(m.id for m in group)
        original = next((m.text for m in group if m.text), "") or ""
        photos = [m.photo for m in group if m.photo]
        if not original.strip() and not photos:
            state[ref] = max_id  # service messages, polls, stickers, videos without text
            save_state(state)
            continue

        link = message_link(entity, group[0].id)
        if original.strip():
            try:
                t0 = time.monotonic()
                # In a thread: a blocking HTTP call must not freeze the Telegram connection.
                result = await asyncio.to_thread(translate, original, cfg, api_key)
                print(f"  translated {len(original)} chars in {time.monotonic() - t0:.0f}s ({result.get('model', '?')})")
                text = build_post(title, link, result["translation"], result["vocabulary"],
                                  original, cfg["show_original"])
            except TranslationFailed as e:
                print(f"  translation failed: {str(e)[:200]}")
                text = build_post(title, link, "⚠️ Could not translate this post.", [],
                                  original, cfg["show_original"])
            await asyncio.sleep(cfg["seconds_between_requests"])
        else:
            text = build_post(title, link, "", [], "", False)

        t0 = time.monotonic()
        await send_post(client, target, text, photos)
        print(f"  sent in {time.monotonic() - t0:.0f}s")
        state[ref] = max_id
        save_state(state)
        sent += 1
    return sent


async def main():
    cfg = load_config()
    state = load_state()
    api_id = int(os.environ["TG_API_ID"])
    api_hash = os.environ["TG_API_HASH"]
    session = os.environ["TG_SESSION"]
    api_key = os.environ["GEMINI_API_KEY"]

    _fernet()  # fail early if STATE_KEY is missing
    cfg["_deadline"] = time.monotonic() + cfg["max_run_seconds"]
    if not cfg["channels"]:
        sys.exit("No channels: set the TG_CHANNELS secret")

    async with TelegramClient(StringSession(session), api_id, api_hash) as client:
        client.parse_mode = "html"  # message.text comes back as HTML
        dialogs_cache = []
        try:
            target = await resolve(client, str(cfg["target"]), dialogs_cache)
        except ValueError as e:
            sys.exit(f"Target chat: {e}")
        budget = cfg["max_posts_per_run"]
        for n, ref in enumerate(cfg["channels"], 1):
            if budget <= 0:
                break
            label = f"channel #{n}"
            try:
                budget -= await process_channel(client, ref, cfg, state, target, api_key,
                                                budget, dialogs_cache, label)
            except RateLimited:
                print("Gemini quota exhausted, will continue next run")
                break
            except ProviderError as e:
                print(f"Gemini error, nothing marked as done: {e}")
                break
            except Unavailable:
                print("Gemini is overloaded right now, nothing marked as done; retrying next run")
                break
            except Exception as e:
                # Only the error type: messages can contain channel names.
                print(f"Error in {label}: {type(e).__name__}")
    save_state(state)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyError as e:
        sys.exit(f"Missing environment variable {e}")
