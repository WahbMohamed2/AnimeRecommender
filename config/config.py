import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3.8-27b")
# Groq's free tier caps output at 1000 tokens/minute; without an explicit
# limit the API estimates the model default and rejects the request.
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add it to your .env file at the project root."
    )
