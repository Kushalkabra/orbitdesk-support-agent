# demo_retry.py — Demonstrates the real verification-failure-and-revision code path.
#
# This script exists because, after the citation-fallback and retry-counter fixes,
# the model's natural outputs no longer reliably fail verification on their own.
# To produce concrete evidence of the verify → revise → verify loop for the
# assignment's required test case #5, we monkeypatch run_generation so its *first*
# call returns a deliberately ungrounded draft (empty citations, made-up claim).
# On any subsequent call the real generation logic runs normally, allowing the
# retry to produce a properly cited answer.

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph import build_graph
from src.nodes.generate import run_generation as real_run_generation


QUESTION = (
    "Our daily dashboard exports stopped appearing at the expected time after "
    "an Admin changed the workspace timezone yesterday. The schedule still "
    "looks active. What should we check, and can the missed export be recovered?"
)

FAKE_DRAFT = (
    "OrbitDesk has a built-in AutoRecover feature that automatically re-queues "
    "any missed export within 30 minutes. Simply navigate to Settings > "
    "AutoRecover and toggle it on. No manual intervention is needed."
)


def main():
    call_count = 0

    def patched_run_generation(state):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            print("\n" + "=" * 70)
            print("[DEMO] First call -- returning deliberately ungrounded draft")
            print("=" * 70)
            return {
                "draft_answer": FAKE_DRAFT,
                "parsed_sources": [],
                "warnings": [],
            }
        print("\n" + "=" * 70)
        print(f"[DEMO] Call #{call_count} -- falling through to real generation")
        print("=" * 70)
        return real_run_generation(state)

    # _log wrapper uses node_fn.__name__ for node_trace entries
    patched_run_generation.__name__ = "run_generation"

    graph = build_graph()

    with patch("src.graph.run_generation", patched_run_generation):
        compiled = build_graph()
        result = compiled.invoke(
            {
                "question": QUESTION,
                "revision_count": 0,
                "node_trace": [],
            }
        )

    node_trace = result.get("node_trace", [])
    final_response = result.get("final_response", {})
    verification_reason = result.get("verification_reason", "(not set)")

    print("\n" + "=" * 70)
    print("[DEMO] RESULTS")
    print("=" * 70)

    print(f"\nNode trace: {node_trace}")
    print(f"\nFirst-attempt verification failure reason: {verification_reason}")
    print(f"\nFinal response:\n{json.dumps(final_response, indent=2)}")

    gen_count = node_trace.count("run_generation")
    ver_count = node_trace.count("run_verification")
    print(f"\nrun_generation appeared {gen_count} time(s)")
    print(f"run_verification appeared {ver_count} time(s)")

    if gen_count >= 2 and ver_count >= 2:
        print("\n[PASS] Retry loop confirmed: generation ran at least twice with verification in between.")
    else:
        print("\n[FAIL] Retry loop NOT observed -- check the trace above.")


if __name__ == "__main__":
    main()
