from __future__ import annotations

import html

import streamlit as st

from main import default_state, run_turn

st.set_page_config(page_title="AutoStream AI Agent", page_icon="AI", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #f6efe5;
        --panel: rgba(255, 250, 245, 0.82);
        --panel-strong: #fff9f2;
        --ink: #1b1a17;
        --muted: #5c554d;
        --accent: #ff6b35;
        --accent-2: #f7b267;
        --teal: #1e8f8f;
        --line: rgba(27, 26, 23, 0.08);
        --shadow: 0 24px 80px rgba(55, 35, 16, 0.12);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 107, 53, 0.22), transparent 30%),
            radial-gradient(circle at top right, rgba(30, 143, 143, 0.16), transparent 28%),
            linear-gradient(180deg, #fff7ef 0%, #f6efe5 48%, #efe4d4 100%);
        color: var(--ink);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero-shell {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.48);
        background:
            linear-gradient(135deg, rgba(255, 117, 77, 0.95), rgba(255, 186, 107, 0.92) 48%, rgba(30, 143, 143, 0.88));
        border-radius: 28px;
        padding: 1.6rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        margin: 0;
        font-size: clamp(1.8rem, 3.6vw, 3.3rem);
        line-height: 1.02;
        color: white;
        white-space: nowrap;
    }

    .hero-copy {
        max-width: 52rem;
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        line-height: 1.6;
        margin-top: 0.85rem;
        margin-bottom: 0;
    }

    .eyebrow {
        display: inline-block;
        padding: 0.38rem 0.7rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.16);
        color: #fff7f0;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.85rem;
    }

    .studio-card {
        background: var(--panel);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.55);
        border-radius: 24px;
        padding: 1.2rem 1.15rem;
        box-shadow: var(--shadow);
        min-height: 100%;
    }

    .chat-wrap {
        background: rgba(255, 249, 242, 0.72);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.52);
        border-radius: 28px;
        padding: 1rem 1rem 0.75rem 1rem;
        box-shadow: var(--shadow);
    }

    .chat-row {
        border-radius: 22px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.85rem;
        border: 1px solid var(--line);
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }

    .chat-row.user {
        background: linear-gradient(135deg, rgba(30, 143, 143, 0.1), rgba(255, 255, 255, 0.72));
    }

    .chat-row.assistant {
        background: #121212;
        border-color: rgba(255, 107, 53, 0.45);
    }

    .chat-avatar {
        width: 2.15rem;
        height: 2.15rem;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        flex: 0 0 auto;
    }

    .chat-row.assistant .chat-avatar {
        background: linear-gradient(135deg, #ff9f1c, #ff7b00);
    }

    .chat-row.user .chat-avatar {
        background: linear-gradient(135deg, #1e8f8f, #146b6b);
    }

    .chat-body {
        line-height: 1.7;
        font-size: 1rem;
        padding-top: 0.05rem;
    }

    .chat-row.assistant .chat-body,
    .chat-row.assistant .chat-body * {
        color: #ffffff !important;
    }

    .chat-row.user .chat-body,
    .chat-row.user .chat-body * {
        color: #161616 !important;
    }

    .signal-card {
        background: rgba(255,255,255,0.7);
        border: 1px solid rgba(27, 26, 23, 0.08);
        border-radius: 20px;
        padding: 0.95rem 1rem;
        margin-bottom: 0.85rem;
    }

    .signal-label {
        color: var(--muted);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }

    .signal-value {
        font-size: 0.98rem;
        line-height: 1.5;
    }

    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #1f8a8a, #145f73);
        color: #fffaf4 !important;
        border: 1px solid rgba(20, 95, 115, 0.85);
        border-radius: 16px;
        font-weight: 700;
        box-shadow: 0 12px 28px rgba(20, 95, 115, 0.18);
    }

    div[data-testid="stButton"] > button p,
    div[data-testid="stButton"] > button span,
    div[data-testid="stButton"] > button div {
        color: #fffaf4 !important;
    }

    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #27a0a0, #1a7087);
        border-color: rgba(20, 95, 115, 0.95);
        color: #ffffff !important;
    }

    div[data-testid="stButton"] > button:hover p,
    div[data-testid="stButton"] > button:hover span,
    div[data-testid="stButton"] > button:hover div {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero-shell">
        <div class="eyebrow">LangGraph Agent Router</div>
        <h1 class="hero-title">AutoStream Studio Concierge</h1>
        <p class="hero-copy">
            A routing-first assistant that can detect intent shifts, remember user context, retrieve product
            knowledge, and only execute tools when the conversation state is actually ready.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

stats = st.columns(3)
for col, title, value, note in [
    (stats[0], "Agent Flow", "Router -> RAG -> Tool", "LangGraph chooses the next step per turn."),
    (stats[1], "Memory", "Cross-turn context", "The agent remembers topic, platform, and pending workflow."),
    (stats[2], "UI", "State-aware prompts", "Quick buttons now react to the live conversation state."),
]:
    with col:
        st.markdown(
            f"""
            <div class="studio-card">
                <div class="signal-label">{title}</div>
                <div class="signal-value"><strong>{value}</strong><br>{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if "agent_state" not in st.session_state:
    st.session_state.agent_state = default_state()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome to AutoStream's routing-first assistant. Ask about plans, support, refunds, exports, "
                "or tell me when you're ready to start and I'll adapt the workflow as the conversation changes."
            ),
        }
    ]

if "prefill_prompt" not in st.session_state:
    st.session_state.prefill_prompt = ""

left_col, right_col = st.columns([1.55, 0.85], gap="large")
state = st.session_state.agent_state

with right_col:
    st.markdown("### Smart Actions")
    smart_actions = state.get("smart_actions") or [
        "Show me pricing",
        "What features does Pro include?",
        "I want to start a trial",
    ]
    for index, action in enumerate(smart_actions):
        if st.button(action, key=f"smart-action-{index}", use_container_width=True):
            st.session_state.prefill_prompt = action

    st.markdown("### Agent State")
    st.markdown(
        f"""
        <div class="signal-card">
            <div class="signal-label">Current Route</div>
            <div class="signal-value">{html.escape(state['route'])}</div>
        </div>
        <div class="signal-card">
            <div class="signal-label">Last Intent</div>
            <div class="signal-value">{html.escape(state['intent'])}</div>
        </div>
        <div class="signal-card">
            <div class="signal-label">Memory Summary</div>
            <div class="signal-value">{html.escape(state['memory']['summary'] or 'No memory yet')}</div>
        </div>
        <div class="signal-card">
            <div class="signal-label">Pending Workflow</div>
            <div class="signal-value">{html.escape(state['memory']['active_workflow'] or 'None')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if state["awaiting_fields"]:
        st.warning("Waiting for: " + ", ".join(state["awaiting_fields"]))
    if state["intent_shift"]:
        st.info("Intent shift detected on the last turn.")

with left_col:
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        role_class = "assistant" if message["role"] == "assistant" else "user"
        avatar = "U" if message["role"] == "user" else "AI"
        content = html.escape(message["content"]).replace("\n", "<br>")
        st.markdown(
            f"""
            <div class="chat-row {role_class}">
                <div class="chat-avatar">{avatar}</div>
                <div class="chat-body">{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    prompt = st.chat_input(
        "Ask about plans, support, captions, refunds, or say you're ready to try Pro"
    )
    if not prompt and st.session_state.prefill_prompt:
        prompt = st.session_state.prefill_prompt
        st.session_state.prefill_prompt = ""

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.agent_state = run_turn(st.session_state.agent_state, prompt)
        reply = st.session_state.agent_state["response"]
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
