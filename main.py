from __future__ import annotations

import re
from typing import Any
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from intent import detect_intent
from llm import invoke_text
from memory import (
    AgentState,
    IntentLabel,
    RouteLabel,
    build_memory_summary,
    default_lead,
    default_memory,
    default_profile,
    merge_topics,
    trim_history,
)
from rag import create_rag, retrieve_context
from tools import mock_lead_capture

PLATFORMS = ["YouTube", "Instagram", "TikTok", "LinkedIn", "Facebook", "X", "Twitch"]


@lru_cache(maxsize=1)
def get_kb():
    return create_rag()


def _find_email(text: str) -> str | None:
    match = re.search(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", text)
    return match.group(1) if match else None


def _find_platform(text: str) -> str | None:
    lowered = text.lower()
    for platform in PLATFORMS:
        aliases = {platform.lower()}
        if platform == "X":
            aliases.update({"twitter", "x/twitter"})
        if any(alias in lowered for alias in aliases):
            return platform
    return None


def _find_name(text: str, only_if_plain: bool = False) -> str | None:
    patterns = [
        r"\bmy name is\s+([A-Za-z][A-Za-z\s'-]{0,40})",
        r"\bi am\s+([A-Za-z][A-Za-z\s'-]{0,40})",
        r"\bi'm\s+([A-Za-z][A-Za-z\s'-]{0,40})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip(" .,!?")

    if only_if_plain:
        plain = text.strip().strip(" .,!?\n\r\t")
        if re.fullmatch(r"[A-Za-z][A-Za-z\s'-]{1,40}", plain):
            return plain
    return None


def _extract_preferred_plan(text: str) -> str | None:
    lowered = text.lower()
    if "pro" in lowered:
        return "Pro"
    if "basic" in lowered:
        return "Basic"
    return None


def _extract_lead_fields(text: str, current_lead: dict[str, Any], awaiting_fields: list[str]) -> dict[str, str | None]:
    extracted = {
        "name": current_lead.get("name"),
        "email": current_lead.get("email"),
        "platform": current_lead.get("platform"),
    }
    extracted["email"] = extracted["email"] or _find_email(text)
    extracted["platform"] = extracted["platform"] or _find_platform(text)
    extracted["name"] = extracted["name"] or _find_name(text, only_if_plain="name" in awaiting_fields)
    return extracted


def _missing_fields(lead: dict[str, Any]) -> list[str]:
    return [field for field in ("name", "email", "platform") if not lead.get(field)]


def _route_for_intent(intent: IntentLabel, awaiting_fields: list[str], lead_complete: bool) -> RouteLabel:
    if intent == "high_intent_lead":
        return "execute_tool" if lead_complete else "respond_direct"
    if intent == "tool_input":
        return "execute_tool" if lead_complete and awaiting_fields else "respond_direct"
    if intent in {"product_pricing_inquiry", "general_support"}:
        return "retrieve_knowledge"
    return "respond_direct"


def _detect_intent_shift(previous_intent: str, new_intent: str, awaiting_fields: list[str]) -> bool:
    if not previous_intent:
        return False
    if previous_intent == new_intent:
        return False
    if awaiting_fields and new_intent in {"product_pricing_inquiry", "general_support", "greeting"}:
        return True
    return previous_intent in {"high_intent_lead", "tool_input"} and new_intent in {
        "product_pricing_inquiry",
        "general_support",
        "greeting",
    }


def _derive_topics(user_input: str, intent: str, lead: dict[str, Any]) -> list[str]:
    topics: list[str] = []
    lowered = user_input.lower()
    if any(word in lowered for word in ("price", "pricing", "plan", "cost")):
        topics.append("pricing")
    if any(word in lowered for word in ("support", "refund", "policy")):
        topics.append("policy")
    if any(word in lowered for word in ("caption", "4k", "export", "feature")):
        topics.append("features")
    if any(word in lowered for word in ("trial", "demo", "sign", "buy", "start")) or intent == "high_intent_lead":
        topics.append("onboarding")
    if lead.get("platform"):
        topics.append(str(lead["platform"]))
    return topics


def _build_follow_up(missing_fields: list[str]) -> str:
    readable = {
        "name": "your name",
        "email": "your email",
        "platform": "your creator platform",
    }
    questions = [readable[field] for field in missing_fields]
    if len(questions) == 1:
        return f"I just need {questions[0]}."
    if len(questions) == 2:
        return f"I just need {questions[0]} and {questions[1]}."
    return "I just need your name, email, and creator platform."


def _smart_actions(state: AgentState) -> list[str]:
    if state["awaiting_fields"]:
        mapping = {
            "name": "My name is Alex",
            "email": "alex@example.com",
            "platform": "I create for YouTube",
        }
        return [mapping[field] for field in state["awaiting_fields"]]

    topics = state["memory"]["recent_topics"]
    profile = state["profile"]
    if "pricing" in topics:
        return [
            f"Which plan fits {profile['platform']} creators?" if profile["platform"] else "Which plan fits frequent posting?",
            "What does Pro include?",
            "What's the refund policy?",
        ]
    if state["intent"] == "greeting":
        return [
            "Show me pricing",
            "What features does Pro include?",
            "I want to start a trial",
        ]
    if state["intent"] == "high_intent_lead":
        return [
            "I want the Pro plan",
            "How quickly can I get started?",
            "Tell me what happens after signup",
        ]
    return [
        "Compare Basic vs Pro",
        "Do I get 24/7 support?",
        "I want to try Pro",
    ]


def _fallback_answer(intent: str, retrieved_context: str, memory_summary: str, missing_fields: list[str], intent_shift: bool) -> str:
    if intent == "greeting":
        return "Hi! I can help with AutoStream pricing, plan fit, support policies, and getting you started."

    if intent == "high_intent_lead":
        prefix = "We can absolutely get you moving." if not intent_shift else "We can jump back into onboarding."
        return f"{prefix} {_build_follow_up(missing_fields)}"

    if intent == "tool_input" and missing_fields:
        return f"Perfect. {_build_follow_up(missing_fields)}"

    if retrieved_context:
        note = "I picked this up from our knowledge base"
        if memory_summary:
            note += f" and your current context ({memory_summary})"
        return f"{note}:\n\n{retrieved_context}"

    return "I can help with pricing, plan details, support policy, and getting your signup details lined up."


def _generate_response(state: AgentState) -> str:
    missing_fields = _missing_fields(state["lead"])
    shift_line = ""
    if state["intent_shift"]:
        shift_line = "I noticed you changed direction, so I paused the previous flow and answered the new request. "

    if state["route"] == "execute_tool" and state["tool_result"]:
        return (
            f"{shift_line}{state['tool_result']} "
            f"You're set, {state['lead']['name']}. I noted your interest in AutoStream"
            f"{' for ' + state['lead']['platform'] if state['lead']['platform'] else ''}."
        ).strip()

    prompt = f"""
You are the AutoStream sales assistant.
Respond naturally, concisely, and with strong continuity across turns.

Intent: {state['intent']}
Route used: {state['route']}
Intent shift detected: {"yes" if state['intent_shift'] else "no"}
Memory summary: {state['memory']['summary'] or "none"}
Recent topics: {", ".join(state['memory']['recent_topics']) or "none"}
Awaiting lead fields: {", ".join(state['awaiting_fields']) or "none"}
Lead captured already: {"yes" if state['lead_captured'] else "no"}
Retrieved knowledge:
{state['retrieved_context'] or "none"}
Tool result:
{state['tool_result'] or "none"}
User message:
{state['user_input']}

Instructions:
- If the user shifted away from a lead-capture flow, acknowledge that briefly and answer the new question.
- If onboarding is active and fields are still missing, ask only for the remaining fields.
- If knowledge was retrieved, answer from it and avoid inventing extra facts.
- If the user is asking plan-fit advice, tie it to any remembered platform.
- Keep the tone helpful and sales-assistant polished.
"""

    generated = invoke_text(prompt, temperature=0.2)
    if generated:
        return generated

    return shift_line + _fallback_answer(
        intent=state["intent"],
        retrieved_context=state["retrieved_context"],
        memory_summary=state["memory"]["summary"],
        missing_fields=missing_fields,
        intent_shift=state["intent_shift"],
    )


def router_node(state: AgentState) -> AgentState:
    user_input = state["user_input"].strip()
    history = trim_history(state["history"] + [{"role": "user", "content": user_input}])

    lead = _extract_lead_fields(user_input, state["lead"], state["awaiting_fields"])
    profile = {
        **state["profile"],
        "name": lead.get("name") or state["profile"].get("name"),
        "email": lead.get("email") or state["profile"].get("email"),
        "platform": lead.get("platform") or state["profile"].get("platform"),
        "preferred_plan": _extract_preferred_plan(user_input) or state["profile"].get("preferred_plan"),
    }
    memory_summary = build_memory_summary(state["memory"], profile, lead)
    intent = detect_intent(user_input, memory_summary=memory_summary, awaiting_fields=state["awaiting_fields"])
    lead_complete = not _missing_fields(lead)
    route = _route_for_intent(intent, state["awaiting_fields"], lead_complete)
    intent_shift = _detect_intent_shift(state["memory"]["last_intent"], intent, state["awaiting_fields"])

    active_workflow = state["memory"]["active_workflow"]
    if intent in {"high_intent_lead", "tool_input"}:
        active_workflow = "lead_capture"
    elif intent_shift and intent in {"product_pricing_inquiry", "general_support", "greeting"}:
        active_workflow = None

    topics = merge_topics(state["memory"]["recent_topics"], _derive_topics(user_input, intent, lead))
    last_goal = {
        "greeting": "start conversation",
        "product_pricing_inquiry": "learn pricing or product details",
        "high_intent_lead": "begin signup or trial",
        "tool_input": "complete lead details",
        "general_support": "get product help",
    }[intent]
    memory = {
        **state["memory"],
        "last_user_goal": last_goal,
        "recent_topics": topics,
        "last_intent": intent,
        "active_workflow": active_workflow,
        "intent_shift_count": state["memory"]["intent_shift_count"] + (1 if intent_shift else 0),
    }

    updated_state = {
        **state,
        "history": history,
        "lead": lead,
        "profile": profile,
        "intent": intent,
        "route": route,
        "memory": memory,
        "intent_shift": intent_shift,
        "retrieved_docs": [],
        "retrieved_context": "",
        "tool_result": None,
        "notes": ["intent_shift" if intent_shift else "intent_stable"],
    }
    updated_state["memory"]["summary"] = build_memory_summary(updated_state["memory"], profile, lead)
    updated_state["awaiting_fields"] = _missing_fields(lead) if active_workflow == "lead_capture" else []
    return updated_state


def retrieve_node(state: AgentState) -> AgentState:
    docs, context = retrieve_context(get_kb(), state["user_input"])
    return {
        **state,
        "retrieved_docs": docs,
        "retrieved_context": context,
    }


def tool_node(state: AgentState) -> AgentState:
    missing_fields = _missing_fields(state["lead"])
    if missing_fields:
        return {
            **state,
            "awaiting_fields": missing_fields,
            "tool_result": None,
        }

    result = mock_lead_capture(
        str(state["lead"]["name"]),
        str(state["lead"]["email"]),
        str(state["lead"]["platform"]),
    )
    return {
        **state,
        "tool_result": result,
        "lead_captured": True,
        "awaiting_fields": [],
        "memory": {
            **state["memory"],
            "active_workflow": None,
        },
    }


def respond_node(state: AgentState) -> AgentState:
    response = _generate_response(state)
    history = trim_history(state["history"] + [{"role": "assistant", "content": response}])
    final_state = {
        **state,
        "response": response,
        "history": history,
    }
    final_state["smart_actions"] = _smart_actions(final_state)
    final_state["memory"]["summary"] = build_memory_summary(final_state["memory"], final_state["profile"], final_state["lead"])
    return final_state


def route_after_router(state: AgentState) -> str:
    return state["route"]


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("retrieve_knowledge", retrieve_node)
    graph.add_node("execute_tool", tool_node)
    graph.add_node("respond_direct", respond_node)
    graph.add_node("respond_after_retrieval", respond_node)
    graph.add_node("respond_after_tool", respond_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "respond_direct": "respond_direct",
            "retrieve_knowledge": "retrieve_knowledge",
            "execute_tool": "execute_tool",
        },
    )
    graph.add_edge("retrieve_knowledge", "respond_after_retrieval")
    graph.add_edge("execute_tool", "respond_after_tool")
    graph.add_edge("respond_direct", END)
    graph.add_edge("respond_after_retrieval", END)
    graph.add_edge("respond_after_tool", END)
    return graph.compile()


@lru_cache(maxsize=1)
def get_agent():
    return build_graph()


def default_state() -> AgentState:
    return {
        "user_input": "",
        "response": "",
        "intent": "greeting",
        "route": "respond_direct",
        "history": [],
        "lead": default_lead(),
        "awaiting_fields": [],
        "lead_captured": False,
        "memory": default_memory(),
        "profile": default_profile(),
        "retrieved_docs": [],
        "retrieved_context": "",
        "tool_result": None,
        "intent_shift": False,
        "smart_actions": [
            "Show me pricing",
            "What features does Pro include?",
            "I want to start a trial",
        ],
        "notes": [],
    }


def run_turn(state: AgentState, user_input: str) -> AgentState:
    payload = {**state, "user_input": user_input}
    return get_agent().invoke(payload)


def chat():
    state = default_state()
    print("AutoStream AI Agent\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bot: Thanks for chatting with AutoStream.")
            break

        state = run_turn(state, user_input)
        print(f"Bot: {state['response']}")


if __name__ == "__main__":
    chat()
