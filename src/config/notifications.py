# notifications utility for sending messages to Discord via webhook
import requests
import threading
from src.config.settings import DISCORD_WEBHOOK_URL
import time

def _post_webhook(message: str, retries: int = 3, delay: float = 2.0):
    for attempt in range(retries):
        try:
            resp = requests.post(
                DISCORD_WEBHOOK_URL,
                json={"content": message},
                timeout=5
            )
            resp.raise_for_status()
            print(f"[discord] sent: {resp.status_code}")
            print("Limit:", resp.headers.get("X-RateLimit-Limit"))
            print("Remaining:", resp.headers.get("X-RateLimit-Remaining"))
            print("Reset After:", resp.headers.get("X-RateLimit-Reset-After"))
            return  
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print(f"[discord] failed after {retries} attempts: {e}")

def notify_discord(message: str):
    if not DISCORD_WEBHOOK_URL or "None" in DISCORD_WEBHOOK_URL:
        return
    threading.Thread(target=_post_webhook, args=(message,), daemon=True).start()



if __name__ == "__main__":
    message = f"{time.strftime('%Y-%m-%d %H:%M:%S')} ---  This is a test notification."

    notify_discord(message)
