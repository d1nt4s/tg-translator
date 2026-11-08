"""Offline tests: no Telegram or Gemini access needed.  Run: python -m pytest tests"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bot  # noqa: E402


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((url, json))
        return self.responses.pop(0)


def gemini_ok(result):
    return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": json.dumps(result)}]}}]})


import pytest


@pytest.fixture(autouse=True)
def no_waits(monkeypatch):
    async def fast_sleep(_):
        return None
    monkeypatch.setattr(bot.asyncio, "sleep", fast_sleep)


CFG = bot.load_config(Path(__file__).resolve().parents[1] / "config.yaml")
GEMINI_CFG = {**CFG, "provider": "gemini", "model": "gemini-flash-latest"}



def test_translate_parses_json(monkeypatch):
    monkeypatch.setattr(bot.time, "sleep", lambda s: None)
    s = FakeSession([FakeResponse(503, {}), gemini_ok({"translation": "<b>Hi</b>", "vocabulary": []})])
    out = bot.translate("<b>Привет</b>", GEMINI_CFG, "key", session=s)
    assert out["translation"] == "<b>Hi</b>"
    assert "gemini-flash-latest:generateContent" in s.calls[0][0]


def test_daily_quota_raises_rate_limited(monkeypatch):
    monkeypatch.setattr(bot.time, "sleep", lambda s: None)
    s = FakeSession([FakeResponse(429, {"error": "GenerateRequestsPerDayPerProjectPerModel"})])
    try:
        bot.translate("x", GEMINI_CFG, "key", session=s)
    except bot.RateLimited:
        return
    raise AssertionError("expected RateLimited")


def test_build_post_has_vocab_link_and_original():
    text = bot.build_post("Example News", "https://t.me/example_news/1", "Rates <b>rose</b>.",
                          [{"en": "rate hike", "ru": "повышение ставки"}], "Ставки выросли.", True)
    assert text.startswith("<b>Example News</b>")
    assert "📚 <b>Vocabulary</b>\n• <b>rate hike</b> — повышение ставки" in text
    assert '<a href="https://t.me/example_news/1">Original</a>' in text
    assert text.endswith("<blockquote expandable>Ставки выросли.</blockquote>")


def test_long_post_drops_original_and_splits():
    long_tr = "\n\n".join(["word " * 150] * 10)  # ~7500 chars
    text = bot.build_post("T", "https://t.me/x/1", long_tr, [], "оригинал " * 300, True)
    assert "blockquote" not in text
    chunks = bot.split_message(text)
    assert len(chunks) >= 2 and all(bot.visible_len(c) <= bot.TEXT_LIMIT for c in chunks)


def test_group_messages_merges_albums():
    m = lambda i, g: SimpleNamespace(id=i, grouped_id=g)
    groups = bot.group_messages([m(1, None), m(2, 7), m(3, 7), m(4, None)])
    assert [[x.id for x in g] for g in groups] == [[1], [2, 3], [4]]


class FakeClient:
    def __init__(self, messages):
        self.messages = messages
        self.sent = []

    async def get_entity(self, ref):
        return SimpleNamespace(id=42, username=ref, title="Example News")

    async def iter_messages(self, entity, limit=None, min_id=0, reverse=False):
        msgs = [m for m in self.messages if m.id > min_id]
        msgs = msgs if reverse else list(reversed(msgs))
        for m in msgs[:limit]:
            yield m

    async def send_message(self, target, text, **kw):
        self.sent.append(("text", text))

    async def send_file(self, target, file, caption=None, **kw):
        self.sent.append(("file", file, caption))


def msg(i, text="", photo=None, grouped_id=None):
    return SimpleNamespace(id=i, text=text, photo=photo, grouped_id=grouped_id)


def test_process_channel_end_to_end(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("STATE_KEY", "test-key")
    monkeypatch.setattr(bot, "STATE_PATH", tmp_path / "state.enc")
    monkeypatch.setattr(bot.time, "sleep", lambda s: None)
    calls = []
    monkeypatch.setattr(bot, "translate",
                        lambda text, cfg, key: calls.append(text) or {"translation": "EN:" + text, "vocabulary": []})
    client = FakeClient([msg(10, "старый"), msg(11, "новость"), msg(12, "", photo="P1", grouped_id=5),
                         msg(13, "альбом", photo="P2", grouped_id=5), msg(14, "")])
    state = {"example_news": 10}
    sent = asyncio.run(bot.process_channel(client, "example_news", CFG, state, "me", "k", 15, []))
    assert sent == 2 and calls == ["новость", "альбом"]
    assert state["example_news"] == 14  # empty message 14 skipped but progress saved
    assert client.sent[0][0] == "text" and "EN:новость" in client.sent[0][1]
    kind, files, caption = client.sent[1]
    assert kind == "file" and files == ["P1", "P2"] and "EN:альбом" in caption
    assert "https://t.me/example_news/12" in caption
    log = capsys.readouterr().out
    assert "example_news" not in log.lower() and "Example News" not in log and "новость" not in log


def test_new_channel_takes_only_initial_posts(monkeypatch, tmp_path):
    monkeypatch.setenv("STATE_KEY", "test-key")
    monkeypatch.setattr(bot, "STATE_PATH", tmp_path / "state.enc")
    monkeypatch.setattr(bot.time, "sleep", lambda s: None)
    monkeypatch.setattr(bot, "translate", lambda t, c, k: {"translation": t, "vocabulary": []})
    client = FakeClient([msg(i, f"пост {i}") for i in range(1, 21)])
    state = {}
    asyncio.run(bot.process_channel(client, "example_news", CFG, state, "me", "k", 15, []))
    assert len(client.sent) == CFG["initial_posts"] and state["example_news"] == 20


def test_private_settings_come_from_env():
    env = {"TG_CHANNELS": "example_news, @another_channel\n-1001234", "TG_TARGET": "My English News"}
    cfg = bot.load_config(Path(__file__).resolve().parents[1] / "config.yaml", env=env)
    assert cfg["channels"] == ["example_news", "another_channel", "-1001234"]
    assert cfg["target"] == "My English News"


def test_public_config_has_no_channels():
    cfg = bot.load_config(Path(__file__).resolve().parents[1] / "config.yaml", env={})
    assert cfg["channels"] == [] and cfg["target"] == "me"


def test_state_is_encrypted(monkeypatch, tmp_path):
    monkeypatch.setenv("STATE_KEY", "secret-passphrase")
    path = tmp_path / "state.enc"
    bot.save_state({"example_news": 123456}, path)
    raw = path.read_bytes()
    assert b"example_news" not in raw and b"123456" not in raw
    assert bot.load_state(path) == {"example_news": 123456}
    monkeypatch.setenv("STATE_KEY", "wrong")
    assert bot.load_state(path) == {}


def test_json_in_code_fence_is_parsed():
    raw = 'Sure!\n```json\n{"translation": "Hi", "vocabulary": []}\n```'
    assert bot.parse_result(raw)["translation"] == "Hi"


def test_key_error_stops_run_without_skipping_posts(monkeypatch, tmp_path):
    monkeypatch.setenv("STATE_KEY", "k")
    monkeypatch.setattr(bot, "STATE_PATH", tmp_path / "state.enc")
    s = FakeSession([FakeResponse(403, {"error": "free tier is not available"})])
    try:
        bot.translate("x", CFG, "key", session=s)
        raise AssertionError("expected ProviderError")
    except bot.ProviderError:
        pass

    def boom(*a):
        raise bot.ProviderError("403")
    monkeypatch.setattr(bot, "translate", boom)
    client = FakeClient([msg(1, "a"), msg(2, "b")])
    state = {"example_news": 0}
    try:
        asyncio.run(bot.process_channel(client, "example_news", CFG, state, "me", "k", 15, []))
    except bot.ProviderError:
        pass
    assert state["example_news"] == 0 and client.sent == []
