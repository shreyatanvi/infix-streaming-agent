from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=2)
def get_llm(temperature: float = 0.2):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return None

    try:
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            api_key=api_key,
        )
    except Exception:
        return None


def invoke_text(prompt: str, temperature: float = 0.2) -> str | None:
    model = get_llm(temperature=temperature)
    if model is None:
        return None

    try:
        response = model.invoke(prompt)
    except Exception:
        return None

    content = getattr(response, "content", None)
    return content.strip() if isinstance(content, str) else None
