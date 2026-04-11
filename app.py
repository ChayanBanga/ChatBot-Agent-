from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import google.generativeai as genai
from typing import Optional, List
from langgraph.graph import StateGraph, END
from typing import TypedDict
from rag import search_knowledge_base
from intent import classify_intent
from tools import mock_lead_capture

# ── CONFIG ──────────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LANGGRAPH STATE ─────────────────────────────────────────
class AgentState(TypedDict):
    user_message: str
    intent: str
    context: str
    response: str
    collecting_lead: bool
    lead_captured: bool
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    conversation_history: List[dict]
    field_needed: str

# ── SESSION STORE ────────────────────────────────────────────
# Stores state per session so multiple users can chat independently
sessions = {}

def get_initial_state() -> AgentState:
    return {
        "user_message": "",
        "intent": "",
        "context": "",
        "response": "",
        "collecting_lead": False,
        "lead_captured": False,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "conversation_history": [],
        "field_needed": ""
    }

# ── NODES ────────────────────────────────────────────────────
def classify_intent_node(state: AgentState) -> AgentState:
    intent = classify_intent(state["user_message"], API_KEY)
    state["intent"] = intent
    return state

def retrieve_context_node(state: AgentState) -> AgentState:
    context = search_knowledge_base(state["user_message"])
    state["context"] = context
    return state

def generate_response_node(state: AgentState) -> AgentState:
    history_text = ""
    for turn in state["conversation_history"][-6:]:
        history_text += f"{turn['role']}: {turn['content']}\n"

    prompt = f"""
You are a friendly sales assistant for AutoStream, a SaaS product that provides
automated video editing tools for content creators.

Conversation so far:
{history_text}

Relevant knowledge base information:
{state['context']}

User's detected intent: {state['intent']}
User's latest message: "{state['user_message']}"

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
    state["response"] = response.text.strip()

    if state["intent"] == "high_intent" and not state["lead_captured"]:
        state["collecting_lead"] = True
        state["response"] += "\n\nTo get you started, I just need a few quick details. What's your name?"
        state["field_needed"] = "name"

    return state

def collect_lead_node(state: AgentState) -> AgentState:
    field = state["field_needed"]
    value = state["user_message"].strip()

    if not value:
        field_prompts = {
            "name": "Could you share your full name?",
            "email": "What's your email address?",
            "platform": "Which platform do you create content on? (YouTube, Instagram, TikTok, etc.)"
        }
        state["response"] = field_prompts.get(field, "Could you please repeat that?")
        return state

    if field == "name":
        state["lead_name"] = value
        state["field_needed"] = "email"
        state["response"] = f"Nice to meet you, {value}! What's your email address?"

    elif field == "email":
        state["lead_email"] = value
        state["field_needed"] = "platform"
        state["response"] = "Almost there! Which creator platform do you use? (YouTube, Instagram, TikTok, etc.)"

    elif field == "platform":
        state["lead_platform"] = value
        state["field_needed"] = "complete"
        result = mock_lead_capture(
            state["lead_name"],
            state["lead_email"],
            state["lead_platform"]
        )
        state["response"] = (
            f"Perfect! I've got everything I need. {result}\n"
            f"Welcome to AutoStream, {state['lead_name']}! "
            f"Our team will reach out to your email shortly."
        )
        state["lead_captured"] = True
        state["collecting_lead"] = False

    return state

def route(state: AgentState) -> str:
    if state["collecting_lead"] and not state["lead_captured"]:
        return "collect_lead"
    return "classify_intent"

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("collect_lead", collect_lead_node)
    graph.add_edge("classify_intent", "retrieve_context")
    graph.add_edge("retrieve_context", "generate_response")
    graph.add_edge("generate_response", END)
    graph.add_edge("collect_lead", END)
    graph.set_conditional_entry_point(route)
    return graph.compile()

agent_graph = build_graph()

# ── API ──────────────────────────────────────────────────────
class MessageRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.post("/chat")
def chat(req: MessageRequest):
    # Get or create session state
    if req.session_id not in sessions:
        sessions[req.session_id] = get_initial_state()

    state = sessions[req.session_id]
    state["user_message"] = req.message
    state["conversation_history"].append({
        "role": "User",
        "content": req.message
    })

    # Run through graph
    state = agent_graph.invoke(state)
    sessions[req.session_id] = state

    state["conversation_history"].append({
        "role": "Agent",
        "content": state["response"]
    })

    return {
        "response": state["response"],
        "intent": state["intent"],
        "lead_captured": state["lead_captured"]
    }