from __future__ import annotations

import os

from openai import OpenAI


def create_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
        api_key=os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"),
    )


client = create_client() if (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")) else None
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
