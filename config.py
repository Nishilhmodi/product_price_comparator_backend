import os
from dotenv import load_dotenv

load_dotenv(override=True)

SERPAPI_KEY  = os.getenv("SERPAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
REDIS_URL    = os.getenv("REDIS_URL")