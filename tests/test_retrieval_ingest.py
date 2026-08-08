import pytest

from src.ingest import _load_kb_chunks, build_index
from src.nodes.retrieve import TOP_K, run_retrieval
from src.nodes.generate import _format_passages, parse_citations


def test_top_k_is_six():
    assert TOP_K == 6


def test_kb_chunks_have_section_title():
    chunks = _load_kb_chunks()
    assert len(chunks) > 0
    kb_003_chunks = [c for c in chunks if c["source_id"] == "KB-003"]
    assert len(kb_003_chunks) > 1
    titles = [c["section_title"] for c in kb_003_chunks]
    assert "Changing the Timezone" in titles
    assert "Other Time-related Behaviour" in titles


def test_retrieval_returns_top_6_with_section_titles():
    state = {"question": "How do I change the workspace timezone?"}
    res = run_retrieval(state)
    retrieved = res["retrieved"]
    assert len(retrieved) == 6
    assert any("section_title" in chunk for chunk in retrieved)


def test_format_passages_includes_section_title():
    retrieved = [
        {
            "source_id": "KB-003",
            "section_title": "Changing the Timezone",
            "passage": "To apply the new timezone...",
            "score": 0.9,
        }
    ]
    formatted = _format_passages(retrieved)
    assert "[KB-003 Changing the Timezone] To apply the new timezone..." in formatted


def test_parse_citations_with_section_title():
    retrieved = [
        {
            "source_id": "KB-003",
            "section_title": "Changing the Timezone",
            "passage": "To apply the new timezone...",
            "score": 0.9,
        }
    ]
    text = "As stated in [KB-003 Changing the Timezone], timezone changes require resaving."
    citations, warnings = parse_citations(text, retrieved)
    assert len(citations) == 1
    assert citations[0]["source_id"] == "KB-003"
    assert len(warnings) == 0


def test_parse_citations_fallback_inferred():
    retrieved = [
        {
            "source_id": "KB-003",
            "section_title": "Changing the Timezone",
            "passage": "To apply the new timezone...",
            "score": 0.85,
        }
    ]
    text = "To change the timezone, navigate to workspace settings and select save schedule."
    citations, warnings = parse_citations(text, retrieved)
    assert len(citations) == 1
    assert citations[0]["source_id"] == "KB-003"
    assert len(warnings) == 1
    assert "citation inferred from top retrieval match" in warnings[0]


def test_check_no_fabricated_urls():
    from src.nodes.verify import check_no_fabricated_urls, run_verification

    ok, reason = check_no_fabricated_urls("Please visit http://orbitdesk-support.com for details.")
    assert ok is False
    assert "fabricated URL" in reason

    ok, reason = check_no_fabricated_urls("Check out www.example.com/docs")
    assert ok is False
    assert "fabricated URL" in reason

    ok, reason = check_no_fabricated_urls("Refer to [KB-003 Changing the Timezone] for steps.")
    assert ok is True
    assert reason == ""

    # Test run_verification fails on fabricated URL before lexical check
    state = {
        "question": "How do I change timezone?",
        "classification": "answerable",
        "draft_answer": "Visit https://orbitdesk.com/help [KB-003] for timezone steps.",
        "parsed_sources": [{"source_id": "KB-003", "passage": "Changing the workspace timezone..."}],
        "retrieved": [{"source_id": "KB-003", "passage": "Changing the workspace timezone...", "score": 0.9}],
        "top_score": 0.9,
    }
    res = run_verification(state)
    assert res["verification_passed"] is False
    assert res["verification_reason"] == "answer contains a fabricated URL not present in source material"
