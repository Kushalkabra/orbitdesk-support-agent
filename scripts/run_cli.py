#!/usr/bin/env python3
"""
CLI entrypoint for running the OrbitDesk support agent on a single question.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph import build_graph

LOG_PATH = ROOT / "logs" / "run_log.jsonl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OrbitDesk support agent.")
    parser.add_argument(
        "question",
        nargs="?",
        help="Support question to answer. If omitted, you will be prompted interactively.",
    )
    parser.add_argument(
        "--question-id",
        dest="question_id",
        help="Optional tag for sample_questions.json ids when demoing (e.g. Q-001).",
    )
    return parser.parse_args()


def _read_question(args: argparse.Namespace) -> str:
    if args.question:
        return args.question.strip()

    question = input("Question: ").strip()
    if not question:
        raise SystemExit("No question provided.")
    return question


def _format_node_trace(node_trace: list[str]) -> str:
    if not node_trace:
        return "Nodes executed: (none recorded)"
    return "Nodes executed: " + " -> ".join(node_trace)


def _append_run_log(
    *,
    question: str,
    node_trace: list[str],
    final_response: dict,
    question_id: str | None,
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "node_trace": node_trace,
        "final_response": final_response,
    }
    if question_id:
        entry["question_id"] = question_id

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()
    question = _read_question(args)

    graph = build_graph()
    result = graph.invoke(
        {
            "question": question,
            "revision_count": 0,
            "node_trace": [],
        }
    )

    final_response = result.get("final_response", {})
    node_trace = result.get("node_trace", [])

    print(json.dumps(final_response, indent=2, ensure_ascii=False))
    print()
    print(_format_node_trace(node_trace))

    _append_run_log(
        question=question,
        node_trace=node_trace,
        final_response=final_response,
        question_id=args.question_id,
    )


if __name__ == "__main__":
    main()
