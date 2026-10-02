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


# ─── Crisis Keywords & Patterns ─────────────────────────────────────────

# 1. Exact strong matches (fastest)
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
    "take my own life",
    "hang myself",
    "slit my wrist",
    "kill me",
    "nobody would miss me",
]

# 2. Flexible Regex patterns (catches variations and indirect phrases)
CRISIS_PATTERNS = [
    re.compile(r"(want|wish|going)\s+(to|i\s+could)\s+(die|disappear(\s+forever)?|end\s+it(\s+all)?|sleep\s+forever|not\s+wake\s+up)", re.IGNORECASE),
    re.compile(r"(can.t|cannot|cant)\s+(go\s+on|do\s+this|take\s+it|keep\s+going|live\s+like\s+this)\s+(anymore|any\s+longer)", re.IGNORECASE),
    re.compile(r"(no|don.t\s+see\s+the|whats\s+the)\s+(point|reason|purpose)\s+(in|to|of)?\s+(living|life|going\s+on)", re.IGNORECASE),
    re.compile(r"(better\s+off|world\s+would\s+be\s+better|everyone\s+would\s+be\s+better)\s+(without\s+me|if\s+i\s+was\s+dead|if\s+i\s+died)", re.IGNORECASE),
    re.compile(r"(hurt|injure|harm)\s+(myself|my\s+self|me)", re.IGNORECASE),
    re.compile(r"(drink|take|swallow)\s+(bleach|poison|too\s+many\s+pills)", re.IGNORECASE),
    re.compile(r"jump\s+(off|in\s+front\s+of)\s+(a\s+bridge|a\s+building|a\s+train|traffic)", re.IGNORECASE),
]

# ─── ML-Based Crisis Classifier (Zero-Shot) ───────────────────────────
_CRISIS_LABELS = [
    "expressing intent to commit suicide or self-harm",
    "experiencing general sadness, stress, or normal conversation",
]

def _ml_check_crisis(text: str, threshold: float = 0.80) -> bool:
    """
    Uses DeBERTa zero-shot classification to detect subtle suicidal intent.
    Only triggers if the model is highly confident (>80%).
    Reuses the severity classifier in memory.
    """
    classifier = _get_topic_classifier()  # Reusing the lazy-loaded DeBERTa model
    if classifier is None:
        return False

    try:
        result = classifier(
            text,
            candidate_labels=_CRISIS_LABELS,
            hypothesis_template="This text is {}.",
            multi_label=False,
        )

        top_label = result["labels"][0]
        top_score = result["scores"][0]

        # Only block if it's explicitly suicidal AND high confidence
        if top_label == _CRISIS_LABELS[0] and top_score >= threshold:
            print(f"  🚨 ML Crisis Check: Detected suicidal intent (confidence: {top_score:.2%})")
            return True

        return False

    except Exception as e:
        print(f"  ⚠️ ML crisis check failed: {e}")
        return False


def check_crisis_keywords(text: str) -> GuardrailResult:
    """
    Detects crisis-level language indicating immediate danger (suicide/self-harm).

    Uses a 3-pass approach:
    1. Exact Keywords: Fast strict matching for obvious danger.
    2. Regex Patterns: Flexible matching for variations (e.g., "wish I could disappear").
    3. ML Classifier: Zero-shot DeBERTa analysis for subtle/indirect cries for help.

    Returns:
        GuardrailResult with blocked=True if crisis language is detected
    """
    text_lower = text.lower().strip()
    matched_flags = []
    
    # Pass 1: Exact Keyword Match
    for keyword in CRISIS_KEYWORDS:
        if keyword in text_lower:
            matched_flags.append(f"keyword:{keyword}")

    # Pass 2: Regex Pattern Match
    if not matched_flags:
        for pattern in CRISIS_PATTERNS:
            if pattern.search(text):
                matched_flags.append(f"pattern:{pattern.pattern}")
                break
                
    # Pass 3: ML Zero-Shot Classifier (only if regex didn't catch it)
    if not matched_flags:
        # We only run ML if the text is somewhat long, to save time on short simple greetings
        if len(text.split()) > 3 and _ml_check_crisis(text):
             matched_flags.append("ml_classifier_suicidal_intent")

    if matched_flags:
        from guardrails.crisis_response import get_crisis_response
        return GuardrailResult(
            blocked=True,
            reason="crisis_detected",
            response=get_crisis_response(crisis_level="immediate"),
            flags=matched_flags,
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


# ─── ML-Based Topic Classifier (Zero-Shot) ───────────────────────────
# Reuses the same DeBERTa model already loaded for severity detection.
# Only called for ambiguous queries that regex can't decide on (~5% of queries).

# Labels for zero-shot classification
_TOPIC_LABELS = [
    "mental health, psychology, or emotional well-being",
    "not related to mental health or medicine",
]

# Lazy-loaded reference to the severity classifier (avoids circular imports)
_topic_classifier = None


def _get_topic_classifier():
    """
    Lazily loads the same zero-shot classifier used for severity detection.
    This avoids loading a duplicate model — shares the DeBERTa instance.
    If the model fails to load, returns None (graceful fallback to regex-only).
    """
    global _topic_classifier
    if _topic_classifier is not None:
        return _topic_classifier

    try:
        from metrics.severity import classifier as severity_classifier
        _topic_classifier = severity_classifier
        print("  ✅ Off-topic ML classifier loaded (reusing severity model)")
    except ImportError:
        print("  ⚠️ Could not load ML classifier for off-topic detection. Using regex-only fallback.")
        _topic_classifier = None

    return _topic_classifier


def _ml_check_off_topic(text: str, threshold: float = 0.65) -> bool:
    """
    Uses DeBERTa zero-shot classification to determine if a query
    is related to mental health.

    Args:
        text: The user's query text
        threshold: Confidence threshold for blocking. Higher = more lenient.
                   0.65 means we block only if the model is >65% sure it's off-topic.

    Returns:
        True if the query is off-topic, False if it's mental-health related.
    """
    classifier = _get_topic_classifier()
    if classifier is None:
        # Model not available — fallback to allowing (benefit of the doubt)
        return False

    try:
        result = classifier(
            text,
            candidate_labels=_TOPIC_LABELS,
            hypothesis_template="This message is about {}.",
            multi_label=False,
        )

        top_label = result["labels"][0]
        top_score = result["scores"][0]

        print(f"  🔍 ML Topic Check: '{top_label}' (confidence: {top_score:.2%})")

        # Block only if the model is confident it's NOT mental health
        if top_label == _TOPIC_LABELS[1] and top_score >= threshold:
            return True

        return False

    except Exception as e:
        print(f"  ⚠️ ML topic check failed: {e}. Allowing query (benefit of doubt).")
        return False


def check_off_topic(text: str) -> GuardrailResult:
    """
    Detects queries that are clearly not related to mental health.

    Uses a three-pass approach:
    1. REGEX FAST-ALLOW: If any mental health keyword is present → allow immediately
    2. REGEX FAST-BLOCK: If off-topic regex patterns match → block immediately
    3. ML CLASSIFIER: For ambiguous queries (no keyword match, no regex match),
       use DeBERTa zero-shot classification to decide.

    This is intentionally lenient — we'd rather let a borderline query
    through than block someone genuinely seeking help.

    Pass 1 & 2 handle ~95% of queries in <2ms.
    Pass 3 (ML) only runs for the ~5% of ambiguous queries (~200-300ms).

    Returns:
        GuardrailResult with blocked=True if query is clearly off-topic
    """
    text_lower = text.lower()

    # Pass 1: REGEX FAST-ALLOW — mental health keyword found → allow immediately
    for keyword in MENTAL_HEALTH_KEYWORDS:
        if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
            return GuardrailResult(blocked=False)

    # Pass 2: REGEX FAST-BLOCK — obvious off-topic pattern → block immediately
    for pattern in OFF_TOPIC_PATTERNS:
        if pattern.search(text):
            print(f"  🚫 Off-topic detected (regex): matched pattern")
            return GuardrailResult(
                blocked=True,
                reason="off_topic",
                response=OFF_TOPIC_RESPONSE,
                flags=[pattern.pattern],
            )

    # Pass 3: ML CLASSIFIER — ambiguous query, let DeBERTa decide
    # This only runs for queries with no mental health keywords AND no off-topic regex match
    print("  🔍 Ambiguous query — running ML topic classifier...")
    if _ml_check_off_topic(text):
        print(f"  🚫 Off-topic detected (ML classifier)")
        return GuardrailResult(
            blocked=True,
            reason="off_topic_ml",
            response=OFF_TOPIC_RESPONSE,
            flags=["ml_classifier"],
        )

    # All passes cleared — allow through (benefit of the doubt)
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


# ─── PII Redaction ───────────────────────────────────────────────────
# Detects and redacts personally identifiable information (PII) from
# user input BEFORE it reaches the LLM, logs, or any downstream system.
# Uses regex — no external libraries needed.

PII_PATTERNS = {
    "email": re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    ),
    "phone": re.compile(
        # Indian phone numbers: 10 digits, optional +91 prefix
        r'(?:\+91[-\.\s]?)?(?:\(?0?\d{2,4}\)?[-\.\s]?)?\d{5}[-\.\s]?\d{5}'
        r'|\b\d{10}\b'
    ),
    "aadhaar": re.compile(
        # Aadhaar: 12 digits in groups of 4
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
    ),
    "credit_card": re.compile(
        # Credit cards: 16 digits in groups of 4
        r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
    ),
}


def redact_pii(text: str) -> tuple[str, list[str]]:
    """
    Scans user input for personally identifiable information and redacts it.

    Detects:
    - Email addresses (e.g., user@gmail.com)
    - Phone numbers (e.g., 9876543210, +91-98765-43210)
    - Aadhaar numbers (e.g., 1234 5678 9012)
    - Credit card numbers (e.g., 4111-1111-1111-1111)

    Args:
        text: The user's input text (already sanitized)

    Returns:
        Tuple of (redacted_text, list_of_pii_types_found)

    Example:
        >>> redact_pii("email me at ravi@gmail.com, phone 9876543210")
        ("email me at [EMAIL REDACTED], phone [PHONE REDACTED]", ["email", "phone"])
    """
    redacted = text
    found_types = []

    for pii_type, pattern in PII_PATTERNS.items():
        if pattern.search(redacted):
            found_types.append(pii_type)
            redacted = pattern.sub(f"[{pii_type.upper()} REDACTED]", redacted)

    return redacted, found_types


# ─── Main Input Guardrail Runner ─────────────────────────────────────

def run_input_guardrails(text: str) -> GuardrailResult:
    """
    Runs all input guardrails in priority order.

    Order matters:
    1. Sanitize input first (preprocessing)
    2. Redact PII (privacy protection — before anything else sees the data)
    3. Crisis check (highest priority — user safety)
    4. Prompt injection check (security)
    5. Off-topic check (relevance — with ML fallback)

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

    # Step 2: Redact PII (privacy — before data reaches LLM or logs)
    clean_text, pii_flags = redact_pii(clean_text)
    if pii_flags:
        print(f"  🔒 GUARDRAIL: PII redacted — types: {pii_flags}")

    # Step 3: Crisis check (highest priority)
    crisis_result = check_crisis_keywords(clean_text)
    if crisis_result.blocked:
        print(f"  🚨 GUARDRAIL: Crisis detected — keywords: {crisis_result.flags}")
        return crisis_result

    # Step 4: Prompt injection check
    injection_result = check_prompt_injection(clean_text)
    if injection_result.blocked:
        print(f"  🛡️ GUARDRAIL: Prompt injection blocked")
        return injection_result

    # Step 5: Off-topic check (regex + ML fallback)
    off_topic_result = check_off_topic(clean_text)
    if off_topic_result.blocked:
        print(f"  🚫 GUARDRAIL: Off-topic query rejected")
        return off_topic_result

    # All checks passed
    return GuardrailResult(blocked=False)
