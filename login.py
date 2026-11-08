"""Run once on your own computer: logs into Telegram and prints TG_SESSION.

    pip install telethon
    python login.py

Telegram will send a login code to your app. The printed session string gives
full access to your account: store it only in GitHub Secrets, never in the repo.
"""

import base64
import os

from telethon.sync import TelegramClient
from telethon.sessions import StringSession

api_id = int(input("api_id (from my.telegram.org): ").strip())
api_hash = input("api_hash: ").strip()

with TelegramClient(StringSession(), api_id, api_hash) as client:
    print("\nYour channels (use @username or the id in config.yaml):")
    for d in client.iter_dialogs():
        if d.is_channel:
            username = getattr(d.entity, "username", None)
            print(f"  {d.id:>16}  {'@' + username if username else '(private)':<28} {d.name}")
    print("\n=== TG_SESSION (copy everything on the next line) ===")
    print(client.session.save())
    print("\n=== STATE_KEY (random key for the encrypted progress file) ===")
    print(base64.urlsafe_b64encode(os.urandom(32)).decode())
