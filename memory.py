from __future__ import annotations

from typing import Any, Literal, TypedDict


IntentLabel = Literal[
    "greeting",
    "product_pricing_inquiry",
    "high_intent_lead",
    "tool_input",
    "general_support",
]

RouteLabel = Literal["respond_direct", "retrieve_knowledge", "execute_tool"]


class LeadState(TypedDict):
    name: str | None
    email: str | None
    platform: str | None


class ProfileMemory(TypedDict):
    name: str | None
    email: str | None
    platform: str | None
    preferred_plan: str | None


class ConversationMemory(TypedDict):
    summary: str
    last_user_goal: str
    recent_topics: list[str]
    last_intent: str
    active_workflow: str | None
    intent_shift_count: int


class RetrievedDoc(TypedDict):
    title: str
    category: str
    content: str


class AgentState(TypedDict):
    user_input: str
    response: str
    intent: IntentLabel
    route: RouteLabel
    history: list[dict[str, str]]
    lead: LeadState
    awaiting_fields: list[str]
    lead_captured: bool
    memory: ConversationMemory
    profile: ProfileMemory
    retrieved_docs: list[RetrievedDoc]
    retrieved_context: str
    tool_result: str | None
    intent_shift: bool
    smart_actions: list[str]
    notes: list[str]


def default_memory() -> ConversationMemory:
    return {
        "summary": "",
        "last_user_goal": "",
        "recent_topics": [],
        "last_intent": "",
        "active_workflow": None,
        "intent_shift_count": 0,
    }


def default_profile() -> ProfileMemory:
    return {
        "name": None,
        "email": None,
        "platform": None,
        "preferred_plan": None,
    }


def default_lead() -> LeadState:
    return {
        "name": None,
        "email": None,
        "platform": None,
    }


def trim_history(history: list[dict[str, str]], limit: int = 12) -> list[dict[str, str]]:
    return history[-limit:]


def merge_topics(existing: list[str], additions: list[str], limit: int = 6) -> list[str]:
    merged: list[str] = []
    for topic in [*existing, *additions]:
        normalized = topic.strip()
        if normalized and normalized not in merged:
            merged.append(normalized)
    return merged[-limit:]


def build_memory_summary(memory: ConversationMemory, profile: ProfileMemory, lead: LeadState) -> str:
    parts: list[str] = []
    if memory["last_user_goal"]:
        parts.append(f"Goal: {memory['last_user_goal']}")
    if memory["recent_topics"]:
        parts.append("Topics: " + ", ".join(memory["recent_topics"]))
    if profile["platform"]:
        parts.append(f"Platform: {profile['platform']}")
    if profile["preferred_plan"]:
        parts.append(f"Preferred plan: {profile['preferred_plan']}")
    if any(lead.values()):
        collected = ", ".join(f"{key}={value}" for key, value in lead.items() if value)
        parts.append(f"Lead info: {collected}")
    return " | ".join(parts)
