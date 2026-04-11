# AutoStream AI Agent

A conversational AI agent for AutoStream, a SaaS video editing platform.
Built as part of the ServiceHive Inflx internship assignment.

---

## How to Run Locally

1. Clone the repository
   git clone https://github.com/YOUR_USERNAME/ChatBot-Agent-.git
   cd YOUR_REPO_NAME

2. Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Add your API key
   Open agent.py and replace YOUR_API_KEY_HERE with your Gemini API key

5. Run the agent
   python agent.py

---

## Architecture Explanation

The agent is built using Python with the Google Gemini 2.5 Flash model as the LLM backbone.
Instead of using LangGraph's full graph abstraction, the state is managed manually using a
Python dictionary that persists across all conversation turns. This keeps the code readable
and easy to debug while still satisfying the core requirement of retaining memory across
5-6 conversation turns.

The architecture has four components:

1. RAG Pipeline (rag.py): Loads a local JSON knowledge base containing AutoStream pricing
and policies. On each user message, it performs keyword-based retrieval and returns only
the relevant chunk to the agent. This keeps the LLM grounded and prevents hallucination.

2. Intent Classifier (intent.py): Sends the user message to Gemini with a strict prompt
that forces a single-word classification — greeting, inquiry, or high_intent. This
determines how the agent responds at each turn.

3. Lead Capture Tool (tools.py): A mock API function that accepts name, email, and platform.
It is only triggered after all three fields are confirmed — never prematurely. The agent
uses a separate Gemini call to extract each field from natural user responses.

4. Agent Loop (agent.py): The main controller. It maintains a state dictionary tracking
conversation history, lead collection status, and collected lead fields. Each turn follows
a fixed pipeline — classify intent, search knowledge base, generate response, check if
lead collection should begin or continue.

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, I would use the WhatsApp Business API provided by Meta.

The steps would be:

1. Set up a Meta Developer account and create a WhatsApp Business App

2. Register a Webhook URL — this is an HTTPS endpoint (built with FastAPI or Flask)
   that Meta will call every time a user sends a message on WhatsApp

3. The webhook receives a POST request containing the user message in JSON format.
   The server extracts the message text and phone number, then passes it to the agent logic

4. The agent processes the message through the same pipeline (intent → RAG → response)
   and returns a reply

5. The server sends the reply back to the user via the WhatsApp API using a POST request
   to Meta's send message endpoint with the phone number and response text

6. Since WhatsApp conversations are stateful per phone number, the state dictionary in
   agent.py would be keyed by phone number so each user has their own isolated session

The key difference from the terminal version is that instead of a while loop reading
input(), each incoming webhook POST request acts as one conversation turn.

---

## Project Structure

autostream-agent/
├── knowledge_base.json   # Local knowledge base for RAG
├── rag.py                # Knowledge retrieval logic
├── intent.py             # Intent classification using Gemini
├── tools.py              # Mock lead capture tool
├── agent.py              # Main agent loop and state management
├── requirements.txt      # Dependencies
└── README.md             # This file