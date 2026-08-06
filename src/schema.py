from typing import Literal

from pydantic import BaseModel, Field

Classification = Literal[
    "answerable",
    "requires_clarification",
    "requires_escalation",
    "out_of_scope",
    "safe_failure",
]


class SourceRef(BaseModel):
    model_config = {"extra": "forbid"}

    source_id: str
    passage: str = Field(min_length=1)


class SupportResponse(BaseModel):
    model_config = {"extra": "forbid"}

    classification: Classification
    answer: str
    sources: list[SourceRef]
    confidence: float = Field(ge=0, le=1)
    requires_human: bool
    reason: str
    clarification_question: str | None = None
    warnings: list[str] = []
