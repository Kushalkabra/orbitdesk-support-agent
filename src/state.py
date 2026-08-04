"""
Shared state object passed between graph nodes.

Using a TypedDict (not a class with methods) on purpose - LangGraph mutates
this by merging partial dicts returned from each node, so it needs to stay
a plain dict-like structure rather than something with behavior attached.
"""

from typing import TypedDict, Literal, Optional

Classification = Literal[
    "answerable",
    "requires_clarification",
    "requires_escalation",
    "out_of_scope",
    "safe_failure",
]


class RetrievedChunk(TypedDict):
    source_id: str  # e.g. "KB-003" or "CASE-1041"
    passage: str
    score: float


class AgentState(TypedDict, total=False):
    question: str

    classification: Optional[Classification]
    triage_reason: str  # why triage picked this label, used in logs + final "reason"

    retrieved: list[RetrievedChunk]
    top_score: float  # highest similarity score from retrieval, used as a confidence proxy

    draft_answer: str
    clarification_question: Optional[str]

    verification_passed: bool
    verification_reason: str
    revision_count: int  # hard-capped at MAX_REVISIONS in graph.py, prevents infinite loop

    final_response: dict  # matches schema.py / output_schema.json before returning to caller
    node_trace: list[str]  # append node names as they run, used for logging + routing tests
