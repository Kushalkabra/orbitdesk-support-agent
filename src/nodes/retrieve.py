"""
Retrieval node: embed the question and return the top-k most similar chunks.
"""

import numpy as np

from src.ingest import build_index
from src.models import get_embedder
from src.state import AgentState, RetrievedChunk

TOP_K = 4

_INDEX = build_index()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0.0 when either norm is zero."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def run_retrieval(state: AgentState) -> dict:
    question = state["question"]
    query_embedding = get_embedder().encode(question, convert_to_numpy=True)
    if query_embedding.ndim == 2:
        query_embedding = query_embedding[0]

    scored: list[tuple[dict, float]] = [
        (chunk, cosine_similarity(query_embedding, chunk["embedding"]))
        for chunk in _INDEX
    ]
    scored.sort(key=lambda item: item[1], reverse=True)

    retrieved: list[RetrievedChunk] = [
        {
            "source_id": chunk["source_id"],
            "passage": chunk["passage"],
            "score": score,
        }
        for chunk, score in scored[:TOP_K]
    ]

    return {
        "retrieved": retrieved,
        "top_score": retrieved[0]["score"] if retrieved else 0.0,
    }
