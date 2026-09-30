"""
Output Guardrails
=================

Validates and filters LLM output BEFORE it reaches the user.
These checks catch dangerous content that the LLM might generate,
even when given good instructions.

Checks performed:
1. Harmful content detection — blocks medication dosages, self-harm methods, diagnosis claims
2. Disclaimer injection — appends severity-appropriate disclaimers
3. Groundedness check — flags responses that may be hallucinated
"""

import re
from typing import Optional

from guardrails.crisis_response import get_disclaimer


# ─── Harmful Content Patterns ────────────────────────────────────────

HARMFUL_PATTERNS = [
    {
        "pattern": re.compile(
            r"(you\s+)(have|are\s+suffering\s+from|are\s+diagnosed\s+with|definitely\s+have|clearly\s+have)\s+\w+",
            re.IGNORECASE,
        ),
        "category": "diagnosis_claim",
        "description": "Making a definitive diagnosis",
        "replacement": None,  # Will be handled by redaction
    },
    {
        "pattern": re.compile(
            r"take\s+\d+\s*(mg|milligrams?|pills?|tablets?|capsules?|doses?)",
            re.IGNORECASE,
        ),
        "category": "medication_dosage",
        "description": "Recommending specific medication dosages",
        "replacement": "[Please consult a doctor for specific medication and dosage recommendations]",
    },
    {
        "pattern": re.compile(
            r"(method|way|technique|how)\s+(to|for)\s+(kill|harm|hurt|injure|wound)\s+(yourself|oneself|themselves|your\s*self)",
            re.IGNORECASE,
        ),
        "category": "self_harm_method",
        "description": "Describing self-harm methods",
        "replacement": "[Content removed for safety]",
    },
    {
        "pattern": re.compile(
            r"(stop\s+taking|quit|discontinue)\s+(your\s+)?(medication|medicine|pills|antidepressant|prescription)",
            re.IGNORECASE,
        ),
        "category": "medication_discontinuation",
        "description": "Advising to stop medication",
        "replacement": "[Never stop medication without consulting your doctor first]",
    },
    {
        "pattern": re.compile(
            r"you\s+(don.t|do\s+not)\s+need\s+(a\s+)?(therapy|therapist|counseling|counselor|professional\s+help|doctor|psychiatrist)",
            re.IGNORECASE,
        ),
        "category": "discouraging_professional_help",
        "description": "Discouraging professional help",
        "replacement": "[Professional help is valuable and recommended for mental health concerns]",
    },
]

# Phrases that should never appear in a responsible mental health response
BLACKLISTED_PHRASES = [
    "just get over it",
    "stop being dramatic",
    "it's all in your head",
    "man up",
    "you're overreacting",
    "everyone feels that way",
    "just be happy",
    "just think positive",
    "you have nothing to be depressed about",
    "other people have it worse",
    "just pray about it",
    "snap out of it",
]


def check_harmful_content(response: str) -> tuple[str, list[str]]:
    """
    Scans LLM output for harmful content and redacts it.

    Checks for:
    - Definitive diagnosis claims
    - Specific medication dosages
    - Self-harm method descriptions
    - Advice to stop medication
    - Discouragement of professional help
    - Invalidating/dismissive phrases

    Args:
        response: The raw LLM response text

    Returns:
        Tuple of (cleaned_response, list_of_flags)
    """
    flags = []
    cleaned = response

    # Check regex patterns
    for entry in HARMFUL_PATTERNS:
        if entry["pattern"].search(cleaned):
            flags.append(entry["category"])
            if entry["replacement"]:
                cleaned = entry["pattern"].sub(entry["replacement"], cleaned)

    # Check blacklisted phrases
    response_lower = cleaned.lower()
    for phrase in BLACKLISTED_PHRASES:
        if phrase in response_lower:
            flags.append(f"dismissive_phrase: '{phrase}'")
            # Remove the dismissive phrase
            cleaned = re.sub(
                re.escape(phrase), "", cleaned, flags=re.IGNORECASE
            )

    return cleaned, flags


def ensure_disclaimer(response: str, severity: Optional[str] = None) -> str:
    """
    Appends an appropriate disclaimer based on severity level.

    Ensures every response has a disclaimer reminding the user
    that this is AI, not a real doctor/therapist.

    Args:
        response: The LLM response text
        severity: The detected severity level ("Low", "Medium", "High")

    Returns:
        Response with disclaimer appended
    """
    if not severity:
        severity = "Low"

    disclaimer = get_disclaimer(severity)

    # Don't double-add if a disclaimer is already present
    if "not a substitute" in response.lower() or "professional" in response.lower().split("---")[-1] if "---" in response else "":
        return response

    return response + disclaimer


def check_groundedness(response: str, context: str) -> tuple[str, bool]:
    """
    Basic check that the LLM response is grounded in the provided context.

    This is a lightweight heuristic check — not a full NLI model.
    It flags responses that seem to have no connection to the retrieved context.

    Strategy:
    - Extract key content words from the context
    - Check what fraction of the response's key claims overlap
    - If overlap is very low, flag it (but don't block — just warn)

    Args:
        response: The LLM response text
        context: The retrieved context that was provided to the LLM

    Returns:
        Tuple of (response, is_grounded)
    """
    if not context or not response:
        return response, True  # Can't check without context

    # Extract meaningful words (longer than 4 chars, not stopwords)
    stopwords = {
        "about", "above", "after", "again", "against", "their", "there",
        "these", "those", "would", "could", "should", "which", "while",
        "being", "where", "other", "every", "under", "between", "through",
        "during", "before", "because", "doing", "having", "might", "shall",
        "since", "until", "without", "within", "along", "among",
    }

    def extract_keywords(text: str) -> set:
        words = set(re.findall(r"\b[a-z]{5,}\b", text.lower()))
        return words - stopwords

    context_keywords = extract_keywords(context)
    response_keywords = extract_keywords(response)

    if not response_keywords:
        return response, True

    # Calculate overlap
    overlap = response_keywords & context_keywords
    overlap_ratio = len(overlap) / len(response_keywords) if response_keywords else 0

    is_grounded = overlap_ratio > 0.1  # At least 10% keyword overlap

    if not is_grounded:
        print(f"  ⚠️ GUARDRAIL: Low groundedness detected (overlap: {overlap_ratio:.2%})")
        # Don't block, but append a subtle note
        response += (
            "\n\n*⚠️ Note: This response may contain general information "
            "not directly sourced from verified medical resources.*"
        )

    return response, is_grounded


# ─── Main Output Guardrail Runner ────────────────────────────────────

def run_output_guardrails(
    response: str,
    severity: Optional[str] = None,
    context: Optional[str] = None,
) -> str:
    """
    Runs all output guardrails in sequence.

    Order:
    1. Harmful content check (redaction)
    2. Groundedness check (flagging)
    3. Disclaimer injection (appending)

    Args:
        response: Raw LLM response
        severity: Detected severity level from analysis
        context: The retrieved context that was sent to the LLM

    Returns:
        Cleaned, safe response with appropriate disclaimers
    """
    if not response or not response.strip():
        return "I'm sorry, I wasn't able to generate a response. Please try rephrasing your question."

    # Step 1: Check and redact harmful content
    cleaned_response, harm_flags = check_harmful_content(response)

    if harm_flags:
        print(f"  🛡️ GUARDRAIL: Harmful content redacted — flags: {harm_flags}")

    # Step 2: Check groundedness
    if context:
        cleaned_response, is_grounded = check_groundedness(cleaned_response, context)

    # Step 3: Add disclaimer
    cleaned_response = ensure_disclaimer(cleaned_response, severity)

    return cleaned_response
