"""
Triage node: classify incoming questions before retrieval.

Deterministic keyword rules run first (out-of-scope guard + vague-question
detection), then a zero-shot classifier handles everything else.
"""

import re

from src.models import get_classifier
from src.state import AgentState, Classification

OUT_OF_SCOPE_KEYWORDS = [
    "refund",
    "billing",
    "subscription cancellation",
    "cancel subscription",
    "cancel my subscription",
    "legal advice",
]

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:the\s+)?(?:supplied\s+)?documentation", re.I),
    re.compile(r"ignore\s+your\s+instructions", re.I),
    re.compile(r"ignore\s+the\s+supplied\s+material", re.I),
    re.compile(r"disregard\s+(?:the\s+)?(?:supplied|provided|your)\s+(?:documentation|instructions|material)", re.I),
    re.compile(r"ignore\s+(?:all\s+)?(?:prior|previous|above)\s+(?:instructions|rules|documentation)", re.I),
]

VAGUE_PHRASE_PATTERNS = [
    re.compile(r"\bit(?:'s| is) not working\b", re.I),
    re.compile(r"\bsync is broken\b", re.I),
    re.compile(r"\bsync is not working\b", re.I),
    re.compile(r"\bdata sync is not working\b", re.I),
]

# Signals that the question names a concrete object, error, or symptom.
SPECIFICITY_PATTERNS = [
    re.compile(r"\b[a-z]+_[a-z][a-z0-9_]*\b", re.I),  # e.g. render_failed, refresh_already_running
    re.compile(r"\b(?:workspace|connection|schedule|dashboard|credential|export)\s+(?:id|name)\b", re.I),
    re.compile(r"\b(?:ws|conn|case|kb)-[a-z0-9-]+\b", re.I),
    re.compile(r"\berror code\b", re.I),
    re.compile(r"\bfailed with\b", re.I),
    re.compile(r"\b(?:reauthorization_required|connector_internal_error|source_refresh_timeout)\b", re.I),
]

ESCALATION_REPEATED_FAILURE_PATTERNS = [
    re.compile(r"\b(?:two|three|\d+)\s+(?:\w+\s+)*(?:runs?|times?|failures?|errors?|attempts?)\s+in\s+a\s+row\b", re.I),
    re.compile(r"\btwice\s+(?:\w+\s+)*in\s+a\s+row\b", re.I),
    re.compile(r"\bconsecutive\b", re.I),
    re.compile(r"\b(?:repeated|repeating)\s+(?:failure|error|fail|issue|crash)s?\b", re.I),
    re.compile(r"\b(?:failure|error|fail|issue|crash)s?\s+repeatedly\b", re.I),
    re.compile(r"\bmultiple\s+(?:runs?|failures?|errors?|times)\b", re.I),
]

ESCALATION_REPETITION_KEYWORDS_PATTERN = re.compile(
    r"\b(?:again|second\s+time|third\s+time|twice|repeatedly|multiple\s+times)\b", re.I
)

ESCALATION_FAILURE_OR_ERROR_PATTERN = re.compile(
    r"\b(?:[a-z]+_[a-z0-9_]+|error\s+code|error|failure|failed|failing|crash|crashed|timeout)\b", re.I
)

ESCALATION_TROUBLESHOOTING_DONE_PATTERNS = [
    re.compile(r"\balready\s+(?:checked|tried|verified|tested|performed|followed|completed)\b", re.I),
    re.compile(
        r"\b(?:checked|verified|tested|tried|followed)\s+(?:the\s+)?(?:dashboard|connections?|destinations?|logs?|docs?|documentation|status)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:tried|followed|performed)\s+(?:the\s+)?(?:documented|troubleshooting|standard)\s+(?:steps|checks|instructions|guides?)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:documented|troubleshooting)\s+(?:steps|checks|instructions)\s+(?:were\s+)?(?:already\s+)?(?:done|performed|completed|tried|checked)\b",
        re.I,
    ),
]

CLASSIFIER_LABELS: list[Classification] = [
    "answerable",
    "requires_clarification",
    "requires_escalation",
    "out_of_scope",
]

SHORT_QUESTION_WORD_LIMIT = 12

CLARIFICATION_QUESTION = (
    "To diagnose the connection issue, please share your workspace ID, connection name or ID, "
    "current connection state, last successful refresh time, latest error code, and whether "
    "manual and scheduled refreshes are both affected."
)


def _normalize(question: str) -> str:
    return question.strip().lower()


def _matches_out_of_scope(question: str) -> tuple[bool, str]:
    normalized = _normalize(question)

    for keyword in OUT_OF_SCOPE_KEYWORDS:
        if keyword in normalized:
            return True, f"Matched out-of-scope keyword: {keyword!r}"

    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(question):
            return True, f"Matched prompt-injection pattern: {pattern.pattern!r}"

    return False, ""


def _matches_escalation_pattern(question: str) -> bool:
    has_repeated_failure = any(
        pattern.search(question) for pattern in ESCALATION_REPEATED_FAILURE_PATTERNS
    ) or (
        bool(ESCALATION_FAILURE_OR_ERROR_PATTERN.search(question))
        and bool(ESCALATION_REPETITION_KEYWORDS_PATTERN.search(question))
    )

    has_troubleshooting_done = any(
        pattern.search(question) for pattern in ESCALATION_TROUBLESHOOTING_DONE_PATTERNS
    )

    return has_repeated_failure and has_troubleshooting_done


def _has_specificity(question: str) -> bool:
    return any(pattern.search(question) for pattern in SPECIFICITY_PATTERNS)


def _is_vague(question: str) -> bool:
    if _has_specificity(question):
        return False

    normalized = _normalize(question)
    word_count = len(normalized.split())

    if word_count <= SHORT_QUESTION_WORD_LIMIT:
        return True

    return any(pattern.search(question) for pattern in VAGUE_PHRASE_PATTERNS)


def _classify_with_model(question: str) -> tuple[Classification, str]:
    classifier = get_classifier()
    result = classifier(question, candidate_labels=CLASSIFIER_LABELS)
    label = result["labels"][0]
    score = result["scores"][0]
    return label, f"Classifier top label: {label} (score={score:.3f})"


def run_triage(state: AgentState) -> dict:
    question = state["question"]

    matched, reason = _matches_out_of_scope(question)
    if matched:
        return {
            "classification": "out_of_scope",
            "triage_reason": reason,
        }

    if _matches_escalation_pattern(question):
        return {
            "classification": "requires_escalation",
            "triage_reason": (
                "Matched escalation pattern: repeated failure after documented checks "
                "were already performed (see KB-008 escalation conditions)."
            ),
        }

    if _is_vague(question):
        return {
            "classification": "requires_clarification",
            "triage_reason": (
                "Question is too short or uses non-specific phrasing (e.g. sync/not working) "
                "without a named object, error code, or concrete symptom."
            ),
            "clarification_question": CLARIFICATION_QUESTION,
        }

    classification, triage_reason = _classify_with_model(question)
    return {
        "classification": classification,
        "triage_reason": triage_reason,
    }
