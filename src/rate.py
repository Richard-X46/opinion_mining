from groq import Groq
import logging
from dotenv import load_dotenv

load_dotenv() 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
client = Groq()




def log_rate_limits(headers) -> None:
    logging.info("── Groq Rate Limits ─────────────────────────────")
    logging.info(
        f"  Requests : {headers.get('x-ratelimit-remaining-requests')} / {headers.get('x-ratelimit-limit-requests')} remaining  (resets in {headers.get('x-ratelimit-reset-requests')})"
    )
    logging.info(
        f"  Tokens   : {headers.get('x-ratelimit-remaining-tokens')} / {headers.get('x-ratelimit-limit-tokens')} remaining  (resets in {headers.get('x-ratelimit-reset-tokens')})"
    )
    logging.info("─────────────────────────────────────────────────")


response = client.chat.completions.with_raw_response.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "your prompt here"}],
)
log_rate_limits(response.headers)

# Then parse normally as usual
completion = response.parse()
content = completion.choices[0].message.content


