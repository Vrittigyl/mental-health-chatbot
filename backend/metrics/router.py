from transformers import pipeline
import re

print("Loading Query Router Model...")
# A very fast, lightweight model for zero-shot classification
router_classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli",
    device=-1
)

# Crisis keywords — checked BEFORE the ML classifier for speed
CRISIS_KEYWORDS = [
    r"\bsuicid\w*\b", r"\bkill\s*(myself|me)\b", r"\bwant\s*to\s*die\b",
    r"\bend\s*(my|this)\s*life\b", r"\bself[\s-]*harm\b", r"\bcut\s*(myself|me)\b",
    r"\bdead\b", r"\bdeath\b", r"\bhang\s*(myself|me)\b", r"\bjump\s*off\b",
    r"\boverdose\b", r"\bno\s*reason\s*to\s*live\b", r"\bbetter\s*off\s*dead\b",
    r"\bdon'?t\s*want\s*to\s*(live|be\s*alive|exist)\b",
    r"\bhurt\s*(myself|me)\b", r"\bslit\b", r"\bbleed\b",
]

CRISIS_RESPONSE = (
    "🚨 It sounds like you may be going through an extremely difficult time. "
    "Please know that you are not alone, and help is available right now.\n\n"
    "📞 **Crisis Helplines:**\n"
    "• **988 Suicide & Crisis Lifeline** (US): Call or text **988**\n"
    "• **Crisis Text Line**: Text **HELLO** to **741741**\n"
    "• **iCall (India)**: **9152987821**\n"
    "• **Vandrevala Foundation (India)**: **1860-2662-345**\n"
    "• **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/\n\n"
    "💙 Please reach out to a professional. You matter, and there are people who want to help."
)

def _is_crisis(query: str) -> bool:
    """Fast keyword check for crisis/self-harm language."""
    query_lower = query.lower()
    for pattern in CRISIS_KEYWORDS:
        if re.search(pattern, query_lower):
            return True
    return False

def route_query(query: str) -> str:
    """
    Classifies the user query into one of four categories:
    1. 'crisis': Contains self-harm or suicidal language — bypass pipeline immediately
    2. 'mental_health': A mental health concern or psychological question
    3. 'greeting': A casual greeting or conversational pleasantry
    4. 'unrelated': An unrelated question about facts, trivia, or general knowledge
    """
    # FAST PATH: Check crisis keywords first (no ML needed)
    if _is_crisis(query):
        return "crisis"

    candidate_labels = [
        "mental health concern or psychological question",
        "casual greeting or conversational pleasantry",
        "an unrelated question about facts, trivia, or general knowledge"
    ]
    
    result = router_classifier(query, candidate_labels=candidate_labels)
    top_label = result['labels'][0]
    
    if top_label == "casual greeting or conversational pleasantry":
        return "greeting"
    elif top_label == "an unrelated question about facts, trivia, or general knowledge":
        return "unrelated"
    else:
        return "mental_health"

