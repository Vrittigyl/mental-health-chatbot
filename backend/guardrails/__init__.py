"""
Guardrails Module
=================

Safety mechanisms for the Mental Health Chatbot.

Provides input and output guardrails that wrap around the LLM pipeline
to ensure safe, responsible, and on-topic responses.

Usage:
    from guardrails import run_input_guardrails, run_output_guardrails

    # Before sending to LLM
    result = run_input_guardrails(user_input)
    if result.blocked:
        return result.response  # Don't send to LLM

    # After getting LLM response
    safe_response = run_output_guardrails(llm_response, severity, context)
"""

from guardrails.input_guardrails import (
    run_input_guardrails,
    GuardrailResult,
    check_crisis_keywords,
    check_prompt_injection,
    check_off_topic,
    sanitize_input,
)
from guardrails.output_guardrails import (
    run_output_guardrails,
    check_harmful_content,
    ensure_disclaimer,
    check_groundedness,
)
from guardrails.crisis_response import (
    get_crisis_response,
    get_disclaimer,
    format_helplines,
)

__all__ = [
    # Input guardrails
    "run_input_guardrails",
    "GuardrailResult",
    "check_crisis_keywords",
    "check_prompt_injection",
    "check_off_topic",
    "sanitize_input",
    # Output guardrails
    "run_output_guardrails",
    "check_harmful_content",
    "ensure_disclaimer",
    "check_groundedness",
    # Crisis response
    "get_crisis_response",
    "get_disclaimer",
    "format_helplines",
]
