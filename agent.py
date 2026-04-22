"""
AutoStream Conversational AI Agent
Built with LangGraph + Groq (llama-3.3-70b-versatile)
"""

import json
import os
import re
from datetime import datetime
from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ── Load knowledge base ────────────────────────────────────────────────────────
KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
with open(KB_PATH, "r") as f:
    KNOWLEDGE_BASE = json.load(f)

# ── LLM setup ─────────────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

# ── State definition ──────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: Optional[str]           # greeting | inquiry | high_intent
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    collection_step: Optional[str]  # asking_name | asking_email | asking_platform | complete


# ── Mock lead capture tool ────────────────────────────────────────────────────
def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Saves lead to leads.json file."""
    lead = {
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": datetime.now().isoformat()
    }

    leads_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leads.json")

    try:
        with open(leads_path, "r") as f:
            leads = json.load(f)
    except FileNotFoundError:
        leads = []

    leads.append(lead)

    with open(leads_path, "w") as f:
        json.dump(leads, f, indent=2)

    print(f"\n🎯 Lead captured and saved to leads.json: {name}, {email}, {platform}\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"


# ── RAG helper ────────────────────────────────────────────────────────────────
def build_knowledge_context() -> str:
    kb = KNOWLEDGE_BASE
    lines = [
        f"Company: {kb['company']}",
        f"Description: {kb['description']}",
        "",
        "=== PRICING PLANS ===",
    ]
    for plan in kb["plans"]:
        lines.append(f"\n{plan['name']} – {plan['price']}")
        for feat in plan["features"]:
            lines.append(f"  • {feat}")

    lines.append("\n=== COMPANY POLICIES ===")
    for policy in kb["policies"]:
        lines.append(f"  • {policy['topic']}: {policy['detail']}")

    lines.append("\n=== FAQs ===")
    for faq in kb["faqs"]:
        lines.append(f"  Q: {faq['question']}")
        lines.append(f"  A: {faq['answer']}")

    return "\n".join(lines)


KNOWLEDGE_CONTEXT = build_knowledge_context()


# ── Hardcoded collection prompts ──────────────────────────────────────────────
COLLECTION_PROMPTS = {
    "asking_name":     "Great choice! To get you started, could you share your full name?",
    "asking_email":    "Thanks {name}! What's your email address?",
    "asking_platform": "Almost done! Which creator platform do you mainly use? (e.g. YouTube, Instagram, TikTok)",
    "invalid_email":   "That doesn't look like a valid email. Could you re-enter it? (e.g. you@gmail.com)",
}


# ── System prompt (greeting / inquiry only) ───────────────────────────────────
def get_system_prompt() -> str:
    return f"""You are an AI sales assistant for AutoStream, a SaaS platform offering automated video editing tools for content creators.

Use ONLY the following knowledge base to answer product/pricing questions:

{KNOWLEDGE_CONTEXT}

RESPONSE STYLE:
- Be friendly, concise, and helpful.
- Never make up features or prices not in the knowledge base.
- NEVER say things like "you can sign up on our website" or "click here to get started".
- If a user seems interested or has chosen a plan, do NOT keep chatting — just confirm their choice briefly.
"""


# ── ENTRY ROUTER (replaces detect_intent as entry point) ─────────────────────
def entry_router(state: AgentState) -> str:
    """
    This runs FIRST on every turn.
    If we are mid-collection, skip intent detection entirely.
    """
    step = state.get("collection_step")
    if step and step not in ("complete", None):
        return "collect_lead_info"
    return "detect_intent"


# ── Intent detection node ─────────────────────────────────────────────────────
def detect_intent(state: AgentState) -> AgentState:
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    prompt = f"""Classify the following user message into exactly one of these intents:
- greeting: casual hello, hi, how are you, general small talk
- inquiry: asking about features, pricing details, policies, comparisons, how things work
- high_intent: user has decided on a plan, wants to buy/sign up/purchase/get started, or shows any buying interest

Examples of HIGH_INTENT (use this liberally — if there is ANY sign the user wants to proceed, choose high_intent):
- "I want to sign up"
- "how can i purchase"
- "let's go with the pro plan"
- "yeah basic is fine for me"
- "basic is fine"
- "ok good"
- "I'll take it"
- "I want to try the pro plan"
- "how do I get started"
- "I want to buy"
- "sounds good, let's do it"
- "ok i'll go with basic"
- "sign me up"
- "that works for me"
- "i'll take the basic plan"

When in doubt between inquiry and high_intent, always choose high_intent.

Message: "{last_user_msg}"

Reply with ONLY one word: greeting, inquiry, or high_intent."""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip().lower()

    if "high_intent" in raw or "high intent" in raw:
        intent = "high_intent"
    elif "inquiry" in raw:
        intent = "inquiry"
    else:
        intent = "greeting"

    return {**state, "intent": intent}


def route_after_intent(state: AgentState) -> str:
    intent = state.get("intent", "greeting")
    if intent == "high_intent":
        return "start_collection"
    return "generate_response"


# ── Lead collection node ──────────────────────────────────────────────────────
def collect_lead_info(state: AgentState) -> AgentState:
    """Read what the user just typed and store it in the right field."""
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content.strip()
            break

    new_state = dict(state)
    step = state.get("collection_step")

    if step == "asking_name":
        new_state["lead_name"] = last_user_msg
        new_state["collection_step"] = "asking_email"

    elif step == "asking_email":
        email_match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", last_user_msg)
        if email_match:
            new_state["lead_email"] = email_match.group()
            new_state["collection_step"] = "asking_platform"
        else:
            new_state["collection_step"] = "invalid_email"

    elif step == "asking_platform":
        new_state["lead_platform"] = last_user_msg
        new_state["collection_step"] = "complete"

    elif step == "invalid_email":
        # User is retrying email
        email_match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", last_user_msg)
        if email_match:
            new_state["lead_email"] = email_match.group()
            new_state["collection_step"] = "asking_platform"
        # else stay on invalid_email

    return new_state


def route_after_collection(state: AgentState) -> str:
    step = state.get("collection_step")
    if step == "complete":
        return "execute_lead_capture"
    return "generate_collection_reply"


# ── Collection reply node (hardcoded questions, no LLM) ──────────────────────
def generate_collection_reply(state: AgentState) -> AgentState:
    step = state.get("collection_step")

    if step == "asking_email":
        text = COLLECTION_PROMPTS["asking_email"].format(name=state.get("lead_name", ""))
    elif step == "asking_platform":
        text = COLLECTION_PROMPTS["asking_platform"]
    elif step == "invalid_email":
        text = COLLECTION_PROMPTS["invalid_email"]
    else:
        text = "Could you please provide that information?"

    return {**state, "messages": [AIMessage(content=text)]}


# ── Response generation node (greeting / inquiry) ─────────────────────────────
def generate_response(state: AgentState) -> AgentState:
    system_prompt = get_system_prompt()
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {**state, "messages": [response]}


# ── Start collection node ─────────────────────────────────────────────────────
def start_collection_node(state: AgentState) -> AgentState:
    text = COLLECTION_PROMPTS["asking_name"]
    return {
        **state,
        "collection_step": "asking_name",
        "messages": [AIMessage(content=text)],
    }


# ── Lead capture execution node ───────────────────────────────────────────────
def execute_lead_capture(state: AgentState) -> AgentState:
    mock_lead_capture(state["lead_name"], state["lead_email"], state["lead_platform"])
    confirmation = AIMessage(
        content=(
            f"🎉 You're all set, {state['lead_name']}!\n"
            f"I've registered your interest in AutoStream's Pro Plan.\n\n"
            f"📧 We'll be in touch at {state['lead_email']} shortly.\n"
            f"Feel free to start your 7-day free trial at autostream.io!\n\n"
            f"Is there anything else I can help you with?"
        )
    )
    return {**state, "messages": [confirmation], "lead_captured": True}


# ── Build the graph ───────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("detect_intent",           detect_intent)
    graph.add_node("start_collection",        start_collection_node)
    graph.add_node("collect_lead_info",       collect_lead_info)
    graph.add_node("generate_collection_reply", generate_collection_reply)
    graph.add_node("generate_response",       generate_response)
    graph.add_node("execute_lead_capture",    execute_lead_capture)

    # Entry: branch BEFORE intent detection if mid-collection
    graph.set_entry_point("detect_intent")
    graph.set_entry_point.__doc__  # just to avoid lint warning

    # We override the entry with a conditional router
    graph = StateGraph(AgentState)
    graph.add_node("detect_intent",             detect_intent)
    graph.add_node("start_collection",          start_collection_node)
    graph.add_node("collect_lead_info",         collect_lead_info)
    graph.add_node("generate_collection_reply", generate_collection_reply)
    graph.add_node("generate_response",         generate_response)
    graph.add_node("execute_lead_capture",      execute_lead_capture)
    graph.add_node("entry_router",              lambda s: s)  # passthrough node

    graph.set_entry_point("entry_router")

    graph.add_conditional_edges(
        "entry_router",
        entry_router,
        {
            "collect_lead_info": "collect_lead_info",
            "detect_intent":     "detect_intent",
        },
    )

    graph.add_conditional_edges(
        "detect_intent",
        route_after_intent,
        {
            "start_collection": "start_collection",
            "generate_response": "generate_response",
        },
    )

    graph.add_edge("start_collection", END)

    graph.add_conditional_edges(
        "collect_lead_info",
        route_after_collection,
        {
            "execute_lead_capture":    "execute_lead_capture",
            "generate_collection_reply": "generate_collection_reply",
        },
    )

    graph.add_edge("generate_collection_reply", END)
    graph.add_edge("generate_response",         END)
    graph.add_edge("execute_lead_capture",      END)

    return graph.compile()


# ── CLI entrypoint ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AutoStream AI Sales Assistant  (powered by Groq)")
    print("  Type 'exit' or 'quit' to end the conversation.")
    print("=" * 60)

    app = build_graph()

    state: AgentState = {
        "messages":       [],
        "intent":         None,
        "lead_name":      None,
        "lead_email":     None,
        "lead_platform":  None,
        "lead_captured":  False,
        "collection_step": None,
    }

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye! 👋")
            break
        if not user_input:
            continue

        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]
        state = app.invoke(state)

        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                print(f"\nAutoStream AI: {msg.content}")
                break

        if state.get("lead_captured"):
            print("\n[Session complete – lead saved to leads.json ✅]")
            break


if __name__ == "__main__":
    main()
