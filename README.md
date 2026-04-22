# AutoStream AI Sales Agent 🤖

Hey! This is my submission for the ServiceHive ML Intern assignment. I built a conversational AI agent for a fictional SaaS company called **AutoStream** — an automated video editing platform for content creators.

The agent can answer product questions, detect when a user wants to sign up, collect their details step by step, and save them as a lead. Pretty cool stuff honestly.

---

## What it does

- Greets users and answers questions about pricing/features using a local knowledge base (RAG)
- Detects when a user has high intent (wants to sign up)
- Collects name → email → platform one by one
- Saves the lead to a `leads.json` file once all details are collected
- Remembers the full conversation across turns using LangGraph state

---

## Tech Stack

- **Python 3.9+**
- **LangGraph** — for building the stateful conversation graph
- **LangChain + Groq** — LLM calls using `llama-3.3-70b-versatile`
- **JSON file** — local knowledge base (RAG) + lead storage

---

## How to Run Locally

### 1. Clone the repo

```bash
https://github.com/Pooja389/AI_agent
cd AI_agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

```bash
cp .env.example .env
```

Open `.env` and paste your Groq API key:
```
GROQ_API_KEY=your_key_here
```

Get a free Groq key at → https://console.groq.com

### 4. Run it

```bash
python agent.py
```

---

## Sample Conversation

```
You: Hi!
AutoStream AI: Hey! Welcome to AutoStream...

You: What plans do you offer?
AutoStream AI: We have two plans:
  • Basic – $29/month: 10 videos, 720p
  • Pro – $79/month: Unlimited videos, 4K, AI captions, 24/7 support

You: I want to try the Pro plan for my YouTube channel
AutoStream AI: Great choice! To get you started, could you share your full name?

You: Alex Johnson
AutoStream AI: Thanks Alex! What's your email address?

You: alex@gmail.com
AutoStream AI: Almost done! Which creator platform do you mainly use?

You: YouTube
🎯 Lead captured and saved to leads.json: Alex Johnson, alex@gmail.com, YouTube

AutoStream AI: 🎉 You're all set, Alex! We'll be in touch at alex@gmail.com shortly.

[Session complete – lead saved to leads.json ✅]
```

---

## Where is the data saved?

Once a lead is captured, it gets saved to `leads.json` in the same folder as `agent.py`. It looks like this:

```json
[
  {
    "name": "Alex Johnson",
    "email": "alex@gmail.com",
    "platform": "YouTube",
    "captured_at": "2026-04-22T14:30:00"
  }
]
```

Every new lead is appended to the list so nothing gets overwritten.

---

## Project Structure

```
autostream-agent/
├── agent.py            # Main agent — all the logic lives here
├── knowledge_base.json # RAG knowledge base (pricing, policies, FAQs)
├── leads.json          # Created automatically after first lead capture
├── requirements.txt    # Dependencies
├── .env.example        # API key template
└── README.md           # This file
```

---

## Architecture Explanation

I used **LangGraph** for this because it lets me model the conversation as an explicit state machine, which makes the logic really clear and easy to debug. Each "node" in the graph does one specific thing, and edges define what happens next based on conditions.

The way state is managed: there's an `AgentState` TypedDict that carries everything — the full message history, detected intent, collected lead fields (`lead_name`, `lead_email`, `lead_platform`), and a `collection_step` cursor that tracks where we are in the lead collection flow. This state is passed through every node on every turn, so the agent always knows the full context of the conversation without needing any external memory store.

The most important design decision was adding an `entry_router` as the first node. Every message hits this router first — if we're mid-collection (e.g. waiting for email), it skips intent detection entirely and goes straight to `collect_lead_info`. This prevents the LLM from misclassifying "alex@gmail.com" as a greeting and breaking the flow. If we're not collecting, it goes to `detect_intent` as normal.

RAG is implemented by loading `knowledge_base.json` at startup, converting it to a structured text block, and injecting it into the system prompt on every LLM call. It's lightweight and works well for a small knowledge base like this — no vector DB needed.

I chose **AutoGen** as an alternative — but LangGraph was the better fit here because the conversation has a very clear linear flow with branching, and LangGraph's conditional edges map directly onto that. AutoGen is better suited for multi-agent collaboration tasks.

---



## Notes

- The agent validates email format with regex before accepting it
- If an invalid email is entered, it asks again instead of crashing
- All collection prompts are hardcoded strings — no LLM needed for those, which saves API calls and avoids hallucination
