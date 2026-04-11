import json

def load_knowledge_base(path="knowledge_base.json"):
    with open(path, "r") as f:
        return json.load(f)

def search_knowledge_base(query: str) -> str:
    kb = load_knowledge_base()
    query = query.lower()

    results = []

    # Check for pricing related queries
    if any(word in query for word in ["price", "pricing", "cost", "plan", "basic", "pro", "how much"]):
        basic = kb["pricing"]["basic_plan"]
        pro = kb["pricing"]["pro_plan"]
        results.append(
            f"Basic Plan: {basic['price']} — {basic['videos_per_month']} videos/month, "
            f"{basic['resolution']} resolution.\n"
            f"Pro Plan: {pro['price']} — {pro['videos_per_month']} videos/month, "
            f"{pro['resolution']} resolution, AI captions included."
        )

    # Check for policy related queries
    if any(word in query for word in ["refund", "cancel", "policy", "return"]):
        results.append(f"Refund Policy: {kb['policies']['refund_policy']}")

    if any(word in query for word in ["support", "help", "contact"]):
        results.append(f"Support Policy: {kb['policies']['support_policy']}")

    # Check for company related queries
    if any(word in query for word in ["autostream", "what is", "about", "company"]):
        results.append(f"About AutoStream: {kb['company']['description']} Target users: {kb['company']['target_users']}")

    if results:
        return "\n".join(results)
    else:
        return "I have information about AutoStream's pricing plans, refund policy, and support options."