"""
Wires the four responsibilities (triage, retrieve, generate, verify) into a
LangGraph StateGraph with conditional routing.

Routing summary:

    triage --out_of_scope-------------------------------------> END
    triage --requires_clarification----------------------------> END
    triage --answerable / requires_escalation--> retrieve --> generate --> verify
    verify --pass------------------------------------------------> END
    verify --fail, revision_count < MAX_REVISIONS----------------> generate (revise)
    verify --fail, revision_count >= MAX_REVISIONS---------------> END (safe_failure)

Node functions live in nodes/ and each one takes AgentState, returns a
partial dict that gets merged into state by LangGraph. Keeping the node
implementations separate from this file so the routing logic is readable
on its own - this file should be small enough to explain in one sitting.
"""

from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.nodes.triage import run_triage
from src.nodes.retrieve import run_retrieval
from src.nodes.generate import run_generation
from src.nodes.verify import run_verification

MAX_REVISIONS = 1  # one retry, then safe_failure - this is the loop guard


def _log(node_fn):
    """Wraps a node so it appends its own name to node_trace, instead of
    every node file having to remember to do that itself."""

    def wrapped(state: AgentState) -> dict:
        update = node_fn(state)
        update["node_trace"] = state.get("node_trace", []) + [node_fn.__name__]
        return update

    return wrapped


def route_after_triage(state: AgentState) -> str:
    classification = state["classification"]
    if classification == "out_of_scope":
        return "finalize_out_of_scope"
    if classification == "requires_clarification":
        return "finalize_clarification"
    # answerable and requires_escalation both need evidence before answering
    return "retrieve"


def route_after_verify(state: AgentState) -> str:
    if state["verification_passed"]:
        return "finalize_ok"
    # Note: run_verification increments revision_count BEFORE this router function runs.
    # On the 1st failure, revision_count becomes 1. With MAX_REVISIONS = 1, checking <= MAX_REVISIONS
    # allows exactly 1 retry (back to generate). On 2nd failure, revision_count becomes 2 (> MAX_REVISIONS),
    # routing to finalize_safe_failure.
    if state.get("revision_count", 0) <= MAX_REVISIONS:
        return "generate"
    return "finalize_safe_failure"


def finalize_out_of_scope(state: AgentState) -> dict:
    return {
        "final_response": {
            "classification": "out_of_scope",
            "answer": (
                "This request is outside the OrbitDesk support knowledge base. "
                "I can't act on billing, legal, or refund requests, or ignore the "
                "supplied documentation."
            ),
            "sources": [],
            "confidence": 1.0,
            "requires_human": True,
            "reason": state.get("triage_reason", "Request falls outside supported topics."),
        }
    }


def finalize_clarification(state: AgentState) -> dict:
    return {
        "final_response": {
            "classification": "requires_clarification",
            "answer": "I need a bit more detail before I can point you to the right steps.",
            "sources": [],
            "confidence": 0.4,
            "requires_human": False,
            "reason": state.get("triage_reason", "Question lacks the object or error needed to route it."),
            "clarification_question": state.get(
                "clarification_question",
                "Could you share the affected object (schedule, connection, dashboard, "
                "or credential) and any error code you saw?",
            ),
        }
    }


def finalize_ok(state: AgentState) -> dict:
    return {"final_response": state["draft_answer_as_schema"]}


def finalize_safe_failure(state: AgentState) -> dict:
    return {
        "final_response": {
            "classification": "safe_failure",
            "answer": (
                "I wasn't able to produce an answer I could verify against the "
                "supplied documentation, so I'm not going to guess."
            ),
            "sources": [],
            "confidence": 0.0,
            "requires_human": True,
            "reason": state.get("verification_reason", "Verification failed after one revision."),
        }
    }


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("triage", _log(run_triage))
    graph.add_node("retrieve", _log(run_retrieval))
    graph.add_node("generate", _log(run_generation))
    graph.add_node("verify", _log(run_verification))
    graph.add_node("finalize_out_of_scope", finalize_out_of_scope)
    graph.add_node("finalize_clarification", finalize_clarification)
    graph.add_node("finalize_ok", finalize_ok)
    graph.add_node("finalize_safe_failure", finalize_safe_failure)

    graph.set_entry_point("triage")

    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "finalize_out_of_scope": "finalize_out_of_scope",
            "finalize_clarification": "finalize_clarification",
            "retrieve": "retrieve",
        },
    )

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")

    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "finalize_ok": "finalize_ok",
            "generate": "generate",  # revision path - revision_count is bumped inside run_verification
            "finalize_safe_failure": "finalize_safe_failure",
        },
    )

    graph.add_edge("finalize_out_of_scope", END)
    graph.add_edge("finalize_clarification", END)
    graph.add_edge("finalize_ok", END)
    graph.add_edge("finalize_safe_failure", END)

    return graph.compile()
