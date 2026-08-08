import pytest

from src.nodes.triage import _matches_escalation_pattern, run_triage


@pytest.mark.parametrize(
    "question",
    [
        "We are seeing render_failed for two runs in a row even though we already checked the dashboard.",
        "The export failed twice in a row. We checked the dashboard/connections/destination.",
        "We got consecutive failures on conn-101 after we tried the documented steps.",
        "We got render_failed for the second time today. We already checked the connections.",
        "connector_internal_error happened again on workspace ws-999. Documented steps were already performed.",
    ],
)
def test_matches_escalation_pattern_positive(question: str):
    assert _matches_escalation_pattern(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "We got render_failed on dashboard dash-01. How do I fix it?",
        "We got render_failed again on dashboard dash-01.",
        "We already checked the dashboard and connections for workspace ws-123.",
        "How do I set up a scheduled export to S3?",
        "The sync is taking longer than expected.",
    ],
)
def test_matches_escalation_pattern_negative(question: str):
    assert _matches_escalation_pattern(question) is False


def test_run_triage_escalation_rule():
    state = {
        "question": "Dashboard dash-42 failed with render_failed twice in a row. We already checked the dashboard.",
        "revision_count": 0,
        "node_trace": [],
    }
    res = run_triage(state)
    assert res["classification"] == "requires_escalation"
    assert "Matched escalation pattern" in res["triage_reason"]
    assert "KB-008" in res["triage_reason"]
