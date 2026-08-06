import json
from pathlib import Path

import pytest

from src.graph import build_graph

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_questions.json"

PIPELINE_NODES = ("run_triage", "run_retrieval", "run_generation", "run_verification")


@pytest.fixture(scope="session")
def sample_questions() -> dict[str, dict]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return {item["question_id"]: item for item in data["questions"]}


@pytest.fixture(scope="session")
def graph():
    return build_graph()


def _run(graph, question: str) -> dict:
    return graph.invoke(
        {
            "question": question,
            "revision_count": 0,
            "node_trace": [],
        }
    )


def _assert_nodes_in_relative_order(node_trace: list[str], expected: tuple[str, ...]) -> None:
    indices = [node_trace.index(name) for name in expected]
    assert indices == sorted(indices), (
        f"Expected {expected} in relative order, got node_trace={node_trace}"
    )


def test_q005_out_of_scope(graph, sample_questions):
    result = _run(graph, sample_questions["Q-005"]["question"])
    final_response = result["final_response"]
    node_trace = result["node_trace"]

    assert final_response["classification"] == "out_of_scope"
    assert "generate" not in node_trace
    assert "run_generation" not in node_trace


def test_q003_requires_clarification(graph, sample_questions):
    result = _run(graph, sample_questions["Q-003"]["question"])
    final_response = result["final_response"]

    assert final_response["classification"] == "requires_clarification"
    assert final_response.get("clarification_question") is not None


@pytest.mark.parametrize("question_id", ["Q-001", "Q-002", "Q-004"])
def test_answerable_questions_run_full_pipeline(graph, sample_questions, question_id):
    result = _run(graph, sample_questions[question_id]["question"])
    final_response = result["final_response"]
    node_trace = result["node_trace"]

    _assert_nodes_in_relative_order(node_trace, PIPELINE_NODES)
    assert final_response["sources"]


def test_verification_loop_guard(monkeypatch, sample_questions):
    def always_fail_verification(state):
        return {
            "verification_passed": False,
            "verification_reason": "forced failure for loop-guard test",
            "revision_count": state.get("revision_count", 0) + 1,
        }

    always_fail_verification.__name__ = "run_verification"
    monkeypatch.setattr("src.graph.run_verification", always_fail_verification)
    graph = build_graph()

    result = _run(graph, sample_questions["Q-002"]["question"])
    final_response = result["final_response"]
    node_trace = result["node_trace"]

    assert final_response["classification"] == "safe_failure"
    assert node_trace.count("run_verification") >= 1
    assert node_trace.count("run_verification") <= 2
    assert node_trace.count("run_generation") <= 2
