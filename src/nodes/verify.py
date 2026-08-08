"""
Verification node: validate citations, lexical grounding, and output schema.
"""

import logging
import re

logger = logging.getLogger(__name__)

from pydantic import ValidationError

from src.schema import SourceRef, SupportResponse
from src.state import AgentState, RetrievedChunk

# A cross-encoder or NLI entailment check would be a stronger grounding signal than
# this word-overlap heuristic, but we keep verification lightweight and explicit here.
STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "you",
    "your",
    "we",
    "our",
    "they",
    "their",
    "i",
    "as",
    "can",
    "could",
    "should",
    "would",
    "may",
    "might",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "not",
    "no",
    "so",
    "than",
    "then",
    "also",
    "into",
    "about",
    "when",
    "where",
    "which",
    "who",
    "what",
    "how",
    "why",
}

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD_PATTERN = re.compile(r"\b[a-z0-9_]+\b", re.I)
CITATION_STRIP = re.compile(r"\[[^\]]+\]")
URL_PATTERN = re.compile(r"https?://|www\.", re.I)

MIN_SIGNIFICANT_WORD_LEN = 2
MIN_SENTENCE_OVERLAP = 0.15
MAX_UNGROUNDED_SENTENCE_RATIO = 0.5


def _failure(state: AgentState, reason: str) -> dict:
    return {
        "verification_passed": False,
        "verification_reason": reason,
        "revision_count": state.get("revision_count", 0) + 1,
    }


def check_citations(
    parsed_sources: list[dict[str, str]] | None,
    retrieved: list[RetrievedChunk],
) -> tuple[bool, str]:
    if not parsed_sources:
        return False, "no valid source citations"

    retrieved_ids = {chunk["source_id"] for chunk in retrieved}
    for source in parsed_sources:
        if source.get("source_id") not in retrieved_ids:
            return False, "no valid source citations"

    return True, ""


def check_no_fabricated_urls(draft_answer: str) -> tuple[bool, str]:
    if URL_PATTERN.search(draft_answer):
        return False, "answer contains a fabricated URL not present in source material"
    return True, ""


def significant_words(text: str) -> set[str]:
    words = WORD_PATTERN.findall(text.lower())
    return {
        word
        for word in words
        if len(word) >= MIN_SIGNIFICANT_WORD_LEN and word not in STOPWORDS
    }


def sentence_word_overlap(sentence: str, passage_words: set[str]) -> float:
    sentence_words = significant_words(sentence)
    if not sentence_words:
        return 1.0
    overlap = sentence_words & passage_words
    return len(overlap) / len(sentence_words)


def check_lexical_grounding(draft_answer: str, retrieved: list[RetrievedChunk]) -> tuple[bool, str]:
    combined_passages = " ".join(chunk["passage"] for chunk in retrieved)
    passage_words = significant_words(combined_passages)

    cleaned_answer = CITATION_STRIP.sub(" ", draft_answer)
    sentences = [part.strip() for part in SENTENCE_SPLIT.split(cleaned_answer) if part.strip()]
    if not sentences:
        sentences = [cleaned_answer.strip()] if cleaned_answer.strip() else []

    scorable_sentences = [sentence for sentence in sentences if significant_words(sentence)]
    if not scorable_sentences:
        return True, ""

    ungrounded = sum(
        1
        for sentence in scorable_sentences
        if sentence_word_overlap(sentence, passage_words) < MIN_SENTENCE_OVERLAP
    )
    if ungrounded / len(scorable_sentences) > MAX_UNGROUNDED_SENTENCE_RATIO:
        return False, "answer not grounded in retrieved evidence"

    return True, ""


def _build_support_response(state: AgentState, parsed_sources: list[dict[str, str]]) -> SupportResponse:
    classification = state["classification"]
    top_score = state.get("top_score", 0.0)
    confidence = min(max(float(top_score), 0.0), 1.0)

    return SupportResponse(
        classification=classification,
        answer=state["draft_answer"],
        sources=[SourceRef(**source) for source in parsed_sources],
        confidence=confidence,
        requires_human=classification == "requires_escalation",
        reason=(
            "Valid source citations and answer text overlaps retrieved evidence "
            f"(retrieval top_score={confidence:.3f})."
        ),
        warnings=state.get("warnings", []),
    )


def run_verification(state: AgentState) -> dict:
    revision = state.get("revision_count", 0)
    parsed_sources = state.get("parsed_sources")
    retrieved = state.get("retrieved", [])
    draft_answer = state.get("draft_answer", "")

    citations_ok, citation_reason = check_citations(parsed_sources, retrieved)
    if not citations_ok:
        logger.info("verify attempt=%d FAILED: %s", revision, citation_reason)
        return _failure(state, citation_reason)

    urls_ok, url_reason = check_no_fabricated_urls(draft_answer)
    if not urls_ok:
        logger.info("verify attempt=%d FAILED: %s", revision, url_reason)
        return _failure(state, url_reason)

    grounded_ok, grounding_reason = check_lexical_grounding(draft_answer, retrieved)
    if not grounded_ok:
        logger.info("verify attempt=%d FAILED: %s", revision, grounding_reason)
        return _failure(state, grounding_reason)

    try:
        response = _build_support_response(state, parsed_sources)
    except ValidationError:
        logger.info("verify attempt=%d FAILED: schema validation", revision)
        return _failure(state, "schema validation failed")

    logger.info("verify attempt=%d PASSED", revision)
    return {
        "verification_passed": True,
        "draft_answer_as_schema": response.model_dump(),
    }
