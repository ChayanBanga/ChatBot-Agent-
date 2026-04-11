import google.generativeai as genai

def classify_intent(user_message: str, api_key: str) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
You are an intent classifier for a SaaS company called AutoStream.

Classify the following user message into exactly ONE of these three intents:
1. greeting — casual hello, hi, how are you, small talk
2. inquiry — asking about product, pricing, features, plans, policies
3. high_intent — user clearly wants to sign up, try, buy, or subscribe

User message: "{user_message}"

Reply with only one word: greeting, inquiry, or high_intent
"""

    response = model.generate_content(prompt)
    intent = response.text.strip().lower()

    # Safety check
    if intent not in ["greeting", "inquiry", "high_intent"]:
        intent = "inquiry"

    return intent