from __future__ import annotations

import os

from openai import OpenAI


def create_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


client = create_client() if os.getenv("OPENAI_API_KEY") else None
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
