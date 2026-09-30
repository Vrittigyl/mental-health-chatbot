"""
Crisis Response Module
======================

Pre-built crisis response templates with helpline numbers.
Provides immediate, empathetic responses when a user is in crisis,
routing them to real human help instead of AI-generated text.
"""

# ─── Helpline Directory ───────────────────────────────────────────────

HELPLINES = {
    "india": [
        {
            "name": "iCall (TISS)",
            "number": "9152987821",
            "hours": "Mon–Sat, 8am–10pm",
            "description": "Professional counselling by trained psychologists",
        },
        {
            "name": "Vandrevala Foundation",
            "number": "1860-2662-345",
            "hours": "24/7",
            "description": "Free, confidential mental health support",
        },
        {
            "name": "AASRA",
            "number": "9820466726",
            "hours": "24/7",
            "description": "Crisis intervention and suicide prevention",
        },
    ],
    "international": [
        {
            "name": "Crisis Text Line",
            "number": "Text HOME to 741741",
            "hours": "24/7",
            "description": "Free crisis counselling via text",
        },
    ],
}


def format_helplines(region: str = "india") -> str:
    """Format helpline information into a readable string."""
    lines = []
    helpline_list = HELPLINES.get(region, HELPLINES["india"])
    for h in helpline_list:
        lines.append(f"  📞 {h['name']}: {h['number']} ({h['hours']})")
        lines.append(f"     {h['description']}")
    return "\n".join(lines)


# ─── Crisis Response Templates ────────────────────────────────────────

CRISIS_RESPONSE_IMMEDIATE = (
    "I hear you, and I want you to know that what you're feeling matters. "
    "You are not alone in this.\n\n"
    "⚠️ **It sounds like you may be in crisis right now.** "
    "Please reach out to a trained professional who can help:\n\n"
    "{helplines}\n\n"
    "💛 A real person can talk to you right now — please call or text one of the numbers above. "
    "They are free, confidential, and available to help.\n\n"
    "I'm an AI and I'm not equipped to provide the kind of support you need right now, "
    "but these professionals are."
)

CRISIS_RESPONSE_HIGH_SEVERITY = (
    "Thank you for sharing that with me. I can sense this is very difficult for you.\n\n"
    "Based on what you've told me, I strongly encourage you to speak with a "
    "mental health professional who can provide personalized support:\n\n"
    "{helplines}\n\n"
    "💛 You deserve support from someone who can truly help. "
    "Please don't hesitate to reach out to any of these services.\n\n"
    "---\n"
    "⚕️ *This is not a substitute for professional medical advice. "
    "If you are in immediate danger, please call emergency services.*"
)


def get_crisis_response(crisis_level: str = "immediate", region: str = "india") -> str:
    """
    Returns an appropriate crisis response message.

    Args:
        crisis_level: "immediate" for active crisis, "high" for high severity
        region: Region for helpline numbers

    Returns:
        Formatted crisis response string
    """
    helplines_text = format_helplines(region)

    if crisis_level == "immediate":
        return CRISIS_RESPONSE_IMMEDIATE.format(helplines=helplines_text)
    else:
        return CRISIS_RESPONSE_HIGH_SEVERITY.format(helplines=helplines_text)


# ─── Disclaimers ──────────────────────────────────────────────────────

DISCLAIMERS = {
    "Low": (
        "\n\n---\n"
        "💡 *Remember: I'm an AI assistant. For ongoing concerns, "
        "consider speaking with a mental health professional.*"
    ),
    "Medium": (
        "\n\n---\n"
        "⚕️ *Important: This information is for educational purposes only and is not "
        "a substitute for professional medical advice. Please consider consulting "
        "a mental health professional for personalized guidance.*"
    ),
    "High": (
        "\n\n---\n"
        "🚨 *If you are in distress, please reach out to a professional immediately.*\n"
        + format_helplines("india")
        + "\n*This AI is not a replacement for professional help.*"
    ),
}


def get_disclaimer(severity: str) -> str:
    """Returns an appropriate disclaimer based on severity level."""
    return DISCLAIMERS.get(severity, DISCLAIMERS["Low"])
