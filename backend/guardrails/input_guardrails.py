"""
Input Guardrails
================

Validates and filters user input BEFORE it reaches the LLM pipeline.
These checks are intentionally deterministic (keyword + regex based)
rather than ML-based, because safety-critical checks must be fast
and predictable.

Checks performed:
1. Crisis keyword detection — immediate escalation for suicidal/self-harm language
2. Prompt injection detection — blocks attempts to hijack the system prompt
3. Off-topic detection — rejects queries unrelated to mental health
4. Input sanitization — cleans and normalizes input text
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ─── Result Data Class ────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    blocked: bool = False
    reason: Optional[str] = None
    response: Optional[str] = None
    flags: list = field(default_factory=list)


# ─── Crisis Keywords ──────────────────────────────────────────────────

CRISIS_KEYWORDS = [
    "kill myself",
    "want to die",
    "end my life",
    "end it all",
    "suicide",
    "suicidal",
    "self-harm",
    "self harm",
    "cut myself",
    "cutting myself",
    "overdose",
    "no reason to live",
    "better off dead",
    "can't go on",
    "cant go on",
    "hurt myself",
    "hurting myself",
    "don't want to live",
    "dont want to live",
    "not worth living",
    "take my own life",
    "hang myself",
    "jump off",
    "slit my wrist",
    "kill me",
    "i want to disappear forever",
    "nobody would miss me",
    "no one would care if i died",
    "planning to end",
    "goodbye forever",
]


def check_crisis_keywords(text: str) -> GuardrailResult:
    """
    Detects crisis-level language indicating immediate danger.

    Uses keyword matching against a curated list of crisis phrases.
    This is intentionally keyword-based (not ML) for reliability —
    a false negative here could be life-threatening.

    Returns:
        GuardrailResult with blocked=True if crisis language is detected
    """
    text_lower = text.lower().strip()
    matched_keywords = []

    for keyword in CRISIS_KEYWORDS:
        if keyword in text_lower:
            matched_keywords.append(keyword)

    if matched_keywords:
        from guardrails.crisis_response import get_crisis_response
        return GuardrailResult(
            blocked=True,
            reason="crisis_detected",
            response=get_crisis_response(crisis_level="immediate"),
            flags=matched_keywords,
        )

    return GuardrailResult(blocked=False)


# ─── Prompt Injection Detection ──────────────────────────────────────

INJECTION_PATTERNS = [
    # "Ignore all previous instructions"
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior|earlier|preceding)\s+(instructions|prompts|rules|context)", re.IGNORECASE),
    # "You are now X" / "Act as X"
    re.compile(r"you\s+are\s+now\s+(a\s+)?(different|new|evil|unrestricted|jailbroken)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(a\s+)?(different|new|evil|unrestricted|jailbroken|DAN)", re.IGNORECASE),
    # "Forget everything"
    re.compile(r"forget\s+(everything|all|your)\s+(you|instructions|rules|training|guidelines)", re.IGNORECASE),
    # System prompt injection
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"<\s*/?system\s*>", re.IGNORECASE),
    # "Pretend you have no restrictions"
    re.compile(r"pretend\s+(you\s+)?(have\s+)?(no|don.t\s+have)\s+(restrictions|rules|guidelines|limitations)", re.IGNORECASE),
    # "Jailbreak" / "DAN mode"
    re.compile(r"\b(jailbreak|DAN\s*mode|developer\s*mode|unrestricted\s*mode)\b", re.IGNORECASE),
    # "Do not follow your rules"
    re.compile(r"do\s+not\s+follow\s+(your|the|any)\s+(rules|instructions|guidelines)", re.IGNORECASE),
    # "Respond without safety"
    re.compile(r"respond\s+without\s+(any\s+)?(safety|restrictions|filters|guardrails)", re.IGNORECASE),
]

INJECTION_RESPONSE = (
    "I'm a mental health support assistant and I can only help with "
    "mental health related questions. I noticed your message contained "
    "instructions that I'm not able to follow. 😊\n\n"
    "If you're going through something difficult, I'm here to listen. "
    "Feel free to share what's on your mind."
)


def check_prompt_injection(text: str) -> GuardrailResult:
    """
    Detects prompt injection attempts.

    Scans for common patterns used to hijack LLM system prompts.
    This prevents adversarial users from manipulating the chatbot
    into giving harmful advice or breaking character.

    Returns:
        GuardrailResult with blocked=True if injection is detected
    """
    matched_patterns = []

    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            matched_patterns.append(pattern.pattern)

    if matched_patterns:
        return GuardrailResult(
            blocked=True,
            reason="prompt_injection_detected",
            response=INJECTION_RESPONSE,
            flags=matched_patterns,
        )

    return GuardrailResult(blocked=False)


# ─── Off-Topic Detection ─────────────────────────────────────────────

# Keywords that strongly indicate mental health relevance
MENTAL_HEALTH_KEYWORDS = [
    "anxious", "anxiety", "depressed", "depression", "stress", "stressed",
    "panic", "panic attack", "worry", "worried", "sad", "sadness",
    "lonely", "loneliness", "grief", "grieving", "trauma", "ptsd",
    "ocd", "bipolar", "schizophrenia", "eating disorder", "anorexia",
    "bulimia", "insomnia", "sleep", "nightmare", "therapy", "therapist",
    "counseling", "counselor", "medication", "antidepressant",
    "mental health", "emotional", "feelings", "feel", "feeling",
    "crying", "cry", "angry", "anger", "frustrated", "frustration",
    "hopeless", "helpless", "overwhelmed", "burnout", "exhausted",
    "relationship", "breakup", "divorce", "abuse", "bullying",
    "self-esteem", "confidence", "motivation", "phobia", "fear",
    "addiction", "substance", "alcohol", "drugs", "recovery",
    "coping", "mindfulness", "meditation", "well-being", "wellness",
    "suicidal", "self-harm", "hurt", "pain", "suffering",
    "happy", "happiness", "joy", "grateful", "gratitude",
    "scared", "terrified", "nervous", "tense", "restless",
    "mood", "moods", "mood swing", "irritable", "numb",
]

# Off-topic patterns that are clearly not mental health
OFF_TOPIC_PATTERNS = [
    re.compile(r"(write|generate|create)\s+(me\s+)?(a\s+)?(\w+\s+)?(code|program|script|essay|story|poem)", re.IGNORECASE),
    re.compile(r"(what|who)\s+(is|was|are|were)\s+the\s+(capital|president|king|queen|ceo)", re.IGNORECASE),
    re.compile(r"(how\s+to|teach\s+me)\s+(cook|code|program|hack|build|make\s+money)", re.IGNORECASE),
    re.compile(r"(weather|stock\s+price|sports\s+score|recipe|math\s+problem)", re.IGNORECASE),
    re.compile(r"(translate|convert|calculate)\s+", re.IGNORECASE),
]

OFF_TOPIC_RESPONSE = (
    "I appreciate your question, but I'm specifically designed to help with "
    "mental health and emotional well-being topics. 🧠💛\n\n"
    "Here are some things I can help with:\n"
    "  • Understanding and managing emotions (anxiety, depression, stress)\n"
    "  • Coping strategies and self-care techniques\n"
    "  • Information about mental health conditions\n"
    "  • Finding professional help and resources\n\n"
    "Feel free to ask me anything related to how you're feeling or your mental health!"
)


def check_off_topic(text: str) -> GuardrailResult:
    """
    Detects queries that are clearly not related to mental health.

    Uses a two-pass approach:
    1. If any mental health keyword is present, allow the query through
    2. If off-topic patterns match AND no mental health keywords, block it

    This is intentionally lenient — we'd rather let a borderline query
    through than block someone genuinely seeking help.

    Returns:
        GuardrailResult with blocked=True if query is clearly off-topic
    """
    text_lower = text.lower()

    # Pass 1: Check for mental health relevance (lenient — any keyword passes)
    for keyword in MENTAL_HEALTH_KEYWORDS:
        # Use word boundaries to avoid false positives (e.g., "numb" in "numbers")
        if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
            return GuardrailResult(blocked=False)

    # Pass 2: Check for clearly off-topic patterns
    for pattern in OFF_TOPIC_PATTERNS:
        if pattern.search(text):
            return GuardrailResult(
                blocked=True,
                reason="off_topic",
                response=OFF_TOPIC_RESPONSE,
                flags=[pattern.pattern],
            )

    # If no mental health keywords found but also not clearly off-topic,
    # let it through — benefit of the doubt
    return GuardrailResult(blocked=False)


# ─── Input Sanitization ──────────────────────────────────────────────

MAX_INPUT_LENGTH = 2000  # characters


def sanitize_input(text: str) -> str:
    """
    Cleans and normalizes user input.

    - Strips leading/trailing whitespace
    - Removes HTML/script tags
    - Normalizes excessive whitespace
    - Truncates extremely long inputs

    Returns:
        Sanitized text string
    """
    # Strip whitespace
    text = text.strip()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove potential script injections
    text = re.sub(r"(javascript|on\w+)\s*[:=]", "", text, flags=re.IGNORECASE)

    # Normalize whitespace (collapse multiple spaces/newlines)
    text = re.sub(r"\s+", " ", text)

    # Truncate if too long
    if len(text) > MAX_INPUT_LENGTH:
        text = text[:MAX_INPUT_LENGTH]

    return text


# ─── Main Input Guardrail Runner ─────────────────────────────────────

def run_input_guardrails(text: str) -> GuardrailResult:
    """
    Runs all input guardrails in priority order.

    Order matters:
    1. Sanitize input first (preprocessing)
    2. Crisis check (highest priority — user safety)
    3. Prompt injection check (security)
    4. Off-topic check (relevance)

    Returns:
        GuardrailResult — if blocked=True, the query should NOT proceed
        to the LLM. The response field contains the message to return.
    """
    # Step 1: Sanitize
    clean_text = sanitize_input(text)

    if not clean_text:
        return GuardrailResult(
            blocked=True,
            reason="empty_input",
            response="It looks like your message was empty. Please share what's on your mind — I'm here to help. 💛",
        )

    # Step 2: Crisis check (highest priority)
    crisis_result = check_crisis_keywords(clean_text)
    if crisis_result.blocked:
        print(f"  🚨 GUARDRAIL: Crisis detected — keywords: {crisis_result.flags}")
        return crisis_result

    # Step 3: Prompt injection check
    injection_result = check_prompt_injection(clean_text)
    if injection_result.blocked:
        print(f"  🛡️ GUARDRAIL: Prompt injection blocked")
        return injection_result

    # Step 4: Off-topic check
    off_topic_result = check_off_topic(clean_text)
    if off_topic_result.blocked:
        print(f"  🚫 GUARDRAIL: Off-topic query rejected")
        return off_topic_result

    # All checks passed
    return GuardrailResult(blocked=False)
