from __future__ import annotations

import re

from llm import invoke_text
from memory import IntentLabel


HIGH_INTENT_PATTERNS = (
    r"\b(sign me up|signup|sign up|start trial|start a trial|free trial|book demo|schedule demo|get started)\b",
    r"\b(i want to try|i want to buy|i want pro|ready to get started|ready to start)\b",
)
GREETING_PATTERNS = (r"\b(hi|hello|hey|good morning|good evening)\b",)
TOOL_INPUT_HINTS = (
    r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
    r"\bmy name is\b",
    r"\bi am\b",
    r"\bi'm\b",
)
PRODUCT_WORDS = ("price", "pricing", "plan", "plans", "cost", "feature", "features", "support", "refund", "policy")
PLATFORM_WORDS = ("youtube", "instagram", "tiktok", "linkedin", "facebook", "twitter", "x", "twitch")


def _heuristic_intent(user_input: str) -> IntentLabel | None:
    text = user_input.lower().strip()
    if not text:
        return "greeting"

    if any(re.search(pattern, text) for pattern in HIGH_INTENT_PATTERNS):
        return "high_intent_lead"

    if any(re.search(pattern, text) for pattern in GREETING_PATTERNS):
        if any(word in text for word in PRODUCT_WORDS):
            return "product_pricing_inquiry"
        return "greeting"

    if any(word in text for word in PRODUCT_WORDS):
        return "product_pricing_inquiry"

    if any(re.search(pattern, text) for pattern in TOOL_INPUT_HINTS):
        return "tool_input"

    plain_name = re.fullmatch(r"[A-Za-z][A-Za-z\s'-]{1,40}", text)
    if plain_name or any(word in text for word in PLATFORM_WORDS):
        return "tool_input"

    return None


def detect_intent(user_input: str, memory_summary: str = "", awaiting_fields: list[str] | None = None) -> IntentLabel:
    awaiting_fields = awaiting_fields or []
    heuristic = _heuristic_intent(user_input)
    if heuristic and heuristic != "tool_input":
        return heuristic

    if heuristic == "tool_input" and awaiting_fields:
        return "tool_input"

    prompt = f"""
Classify the user message into exactly one label:
- greeting
- product_pricing_inquiry
- high_intent_lead
- tool_input
- general_support

Definitions:
- greeting: casual hello or opening small talk
- product_pricing_inquiry: asks about plans, pricing, features, support, refunds, exports, captions, or product details
- high_intent_lead: indicates readiness to sign up, trial, demo, buy, onboard, or get started
- tool_input: provides lead-capture details like name, email, or creator platform
- general_support: everything else

Current memory summary: {memory_summary or "none"}
Awaiting fields: {", ".join(awaiting_fields) if awaiting_fields else "none"}
User message: {user_input}

Return only the label.
"""

    result = invoke_text(prompt, temperature=0)
    if result:
        label = result.strip().lower().split()[0]
        if label in {"greeting", "product_pricing_inquiry", "high_intent_lead", "tool_input", "general_support"}:
            return label  # type: ignore[return-value]

    if heuristic:
        return heuristic
    if awaiting_fields:
        return "tool_input"
    return "general_support"
