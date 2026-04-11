# ============================================================
# AGENT FROM SCRATCH (No Framework)
# ============================================================
# This is the original agent built from scratch without any
# external AI framework like LangChain or LangGraph.
# It demonstrates manual implementation of:
# - State management using a Python dictionary
# - Intent detection via direct Gemini API calls
# - RAG pipeline using keyword-based retrieval
# - Tool calling with premature trigger prevention
# - Multi-turn conversation memory
# ============================================================

import os
import google.generativeai as genai
from rag import search_knowledge_base
from intent import classify_intent
from tools import mock_lead_capture

# ── CONFIG ──────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# ── STATE ────────────────────────────────────────────────
state = {
    "intent_history": [],
    "lead_info": {
        "name": None,
        "email": None,
        "platform": None
    },
    "lead_captured": False,
    "collecting_lead": False,
    "conversation_history": []
}

# ── LEAD COLLECTION LOGIC ────────────────────────────────
def get_missing_lead_field() -> str:
    lead = state["lead_info"]
    if not lead["name"]:
        return "name"
    if not lead["email"]:
        return "email"
    if not lead["platform"]:
        return "platform"
    return "complete"

def extract_lead_info(user_message: str, field_needed: str) -> str:
    prompt = f"""
Extract the {field_needed} from this message. 
Return only the extracted value, nothing else.
If you cannot find a clear {field_needed}, return NONE.

Message: "{user_message}"
"""
    response = model.generate_content(prompt)
    value = response.text.strip()
    if value.upper() == "NONE" or not value:
        return None
    return value

# ── RESPONSE GENERATOR ───────────────────────────────────
def generate_response(user_message: str, context: str, intent: str) -> str:
    history_text = ""
    for turn in state["conversation_history"][-6:]:
        history_text += f"{turn['role']}: {turn['content']}\n"

    prompt = f"""
You are a friendly sales assistant for AutoStream, a SaaS product that provides 
automated video editing tools for content creators.

Conversation so far:
{history_text}

Relevant knowledge base information:
{context}

User's detected intent: {intent}
User's latest message: "{user_message}"

Instructions:
- Be concise and helpful
- If intent is greeting, greet warmly and ask how you can help
- If intent is inquiry, answer using the knowledge base information provided
- If intent is high_intent, express excitement and tell them you'd love to get them started
- Never make up information not in the knowledge base
- Keep responses under 4 sentences

Your response:
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# ── MAIN AGENT LOOP ──────────────────────────────────────
def run_agent():
    print("AutoStream AI Assistant (Scratch Version)")
    print("=" * 40)
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        state["conversation_history"].append({
            "role": "User",
            "content": user_input
        })

        if state["collecting_lead"] and not state["lead_captured"]:
            field_needed = get_missing_lead_field()

            if field_needed != "complete":
                extracted = extract_lead_info(user_input, field_needed)
                if extracted:
                    state["lead_info"][field_needed] = extracted
                    field_needed = get_missing_lead_field()

                if field_needed == "complete":
                    result = mock_lead_capture(
                        state["lead_info"]["name"],
                        state["lead_info"]["email"],
                        state["lead_info"]["platform"]
                    )
                    response = (
                        f"Perfect! I've got everything I need. {result}\n"
                        f"Welcome to AutoStream, {state['lead_info']['name']}! "
                        f"Our team will reach out to your email shortly."
                    )
                    state["lead_captured"] = True
                    state["collecting_lead"] = False
                else:
                    field_prompts = {
                        "name": "Great! Could you share your name?",
                        "email": f"Nice to meet you, {state['lead_info']['name']}! What's your email address?",
                        "platform": f"Almost there! Which creator platform do you use? (YouTube, Instagram, TikTok, etc.)"
                    }
                    response = field_prompts[field_needed]

                print(f"Agent: {response}\n")
                state["conversation_history"].append({
                    "role": "Agent",
                    "content": response
                })
                continue

        intent = classify_intent(user_input, API_KEY)
        state["intent_history"].append(intent)
        context = search_knowledge_base(user_input)
        response = generate_response(user_input, context, intent)

        if intent == "high_intent" and not state["lead_captured"]:
            state["collecting_lead"] = True
            response += "\n\nTo get you started, I just need a few quick details. What's your name?"

        print(f"Agent: {response}\n")
        state["conversation_history"].append({
            "role": "Agent",
            "content": response
        })

if __name__ == "__main__":
    run_agent()