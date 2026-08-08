# OrbitDesk Support Agent

A local-first, graph-orchestrated support agent that answers questions about a fictional
product (OrbitDesk) using only a supplied knowledge base and resolved-case history. Built
for the AI Engineer Internship screening assignment.

Runs entirely on local Hugging Face models via LangGraph. No hosted LLM APIs are used
anywhere in the pipeline.

## Architecture

The graph has four core responsibilities, each implemented as one or more LangGraph nodes,
wired together with conditional routing over a shared typed state (`src/state.py`):

```
triage --out_of_scope-------------------------------------> END (safe out-of-scope response)
triage --requires_clarification----------------------------> END (clarification question)
triage --answerable / requires_escalation--> retrieve -> generate -> verify
verify --pass------------------------------------------------> END (final answer)
verify --fail, revision_count < 1-----------------------------> generate (revise with feedback)
verify --fail, revision_count >= 1-----------------------------> END (safe_failure)
```

Note: answerable and requires_escalation intentionally share the same retrieve → generate → verify pipeline rather than routing to separate end states. An escalation-worthy question still benefits from a grounded, cited answer describing what to do next — the distinction is carried in the final response's requires_human field (set to true for escalation), not through a separate graph path. finalize_ok handles both outcomes; only classification and requires_human differ.

**Triage** — two-stage classification. Deterministic keyword/regex rules run first and
catch three cases directly, without any model call: out-of-scope requests (refunds,
billing, legal advice, or attempts to override the supplied instructions — a
prompt-injection defense), vague questions lacking a named object or error code (routed to
clarification), and a specific escalation pattern (repeated/consecutive failures mentioned
alongside evidence that documented troubleshooting was already attempted, per KB-008's
explicit escalation conditions). Anything not caught by these rules falls through to a
zero-shot classifier (`facebook/bart-large-mnli`).

**Retrieval** — every knowledge-base document is split into chunks by markdown `##`
section, and every resolved case becomes one chunk, each embedded once at startup with
`sentence-transformers/all-MiniLM-L6-v2`. The incoming question is embedded and compared
against all chunks via cosine similarity (plain numpy, no vector database — the corpus is
small enough that a vector DB adds complexity without benefit here). Top 6 chunks are
returned along with their similarity scores, which downstream nodes use as a confidence
signal.

**Generation** — a local instruction-tuned LLM (`Qwen/Qwen2.5-1.5B-Instruct`) is prompted
with only the retrieved passages and told explicitly to cite its sources with bracketed
labels, never fabricate URLs, and say plainly when the evidence doesn't answer the
question. Citations are parsed out of the model's raw output; if the model doesn't include
an explicit citation but a retrieved chunk clears a similarity threshold, the top match is
attached as an inferred citation and a warning is added to the response noting that the
citation wasn't explicitly stated — this keeps a well-grounded answer from being thrown
away over a formatting slip, while being transparent about the difference.

**Verification** — checks the draft answer against three gates: valid, non-fabricated
source citations; no fabricated URLs (the source material contains none, so any URL in the
output is a hallucination by definition); and lexical grounding, a heuristic that checks
whether the answer's claims actually overlap with the retrieved evidence. If any check
fails and no revision has been attempted yet, the failure reason is fed back into a second
generation attempt. If it fails twice, the graph returns a `safe_failure` response rather
than guessing.

**Loop guard** — `revision_count` in shared state is capped at 1 revision. The routing
condition explicitly checks this count before allowing a second pass through generation,
preventing an infinite retry loop.

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
python scripts/download_models.py   # one-time, requires network
```

After the download step, models are cached locally (`~/.cache/huggingface/hub`) and the
application runs fully offline.

## Running

```bash
python scripts/run_cli.py --question-id Q-001 "Our daily dashboard exports stopped appearing..."
```

Or interactively:
```bash
python scripts/run_cli.py
```

Every run appends a full record (question, node trace, final response, timestamp) to
`logs/run_log.jsonl`. A snapshot of the final sample runs is committed at
`docs/sample_outputs.jsonl`.

## Models used

| Model | Revision | Purpose | Approx. load time |
|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | Embedding for retrieval | ~6-9s online / <0.5s cached offline |
| `facebook/bart-large-mnli` | `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce` | Zero-shot triage fallback classification | ~2s online / <1s cached offline |
| `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | Answer generation | ~2.5-4s online / <1s cached offline |

Generation latency (CPU, no GPU acceleration used for the LLM in the final runs) ranged
from roughly 48 seconds to just over 2 minutes per call, depending on answer length. This
directly shaped the choice of a 1.5B model over a larger one — a bigger model would have
made the test suite and live demo impractically slow on this hardware. Retrieval and
triage add well under a second combined once models are loaded.

## Hardware used

- CPU: AMD Ryzen 5 5600H with Radeon Graphics
- RAM: ~7.5 GB (7,522 MB total)
- GPU available: NVIDIA GTX 1650 (4GB VRAM) — used selectively; kept the embedding and
  classification models off-GPU to leave headroom, since 4GB is tight once the generation
  model is loaded alongside them
- CPU-only inference used for the LLM in final runs (see model table above for latency
  figures)
- This is a genuinely constrained machine for a 1.5B-parameter local model plus two
  smaller models running concurrently, which directly shaped the model-size and
  CPU/GPU-placement decisions documented above.

## Sample test cases

All five required cases are demonstrated:

1. **Directly answerable** — Q-002 ("Can a Viewer create an API credential?"). Single-source
   answer from KB-002 / CASE-1058, correctly says no.
2. **Requires two documents** — Q-004 (repeated `render_failed` after documented checks). Triage's deterministic rule classifies this as `requires_escalation`, which flows through the same retrieve → generate → verify pipeline as an answerable question, but produces a response with `requires_human: true`. The underlying evidence spans CASE-1103 and KB-004.
3. **Ambiguous, needs clarification** — Q-003 ("sync is not working"). No object, error
   code, or symptom given; triage asks for the specific fields KB-006 requires before
   diagnosis is possible.
4. **Out of scope** — Q-005 (refund request combined with an instruction to ignore the
   supplied documentation). Caught by the deterministic out-of-scope/prompt-injection rule
   before any model call.
5. **Fails verification, then revises** — demonstrated via `scripts/demo_retry.py`, which
   deliberately forces an ungrounded first draft to trigger a real verification failure,
   then lets the second generation attempt (with the failure reason fed back into the
   prompt) run normally and pass. Real user questions did not reliably fail verification
   naturally after citation-fallback and retry-counter fixes were made, so this script
   exists specifically to demonstrate that code path with a real graph execution rather
   than a mocked one.

An automated routing test suite (`tests/test_routing.py`, `tests/test_triage.py`,
`tests/test_retrieval_ingest.py`) asserts on `node_trace` and `classification` rather than
exact model wording — for example, confirming Q-005 never reaches the generation node, and
that a forced double verification failure terminates in `safe_failure` rather than looping.

## Known limitations

- The lexical-overlap grounding check in verification is a heuristic, not a rigorous
  entailment check. It catches obviously ungrounded claims but isn't as reliable as a
  proper NLI or cross-encoder verifier would be.
- Generation latency on CPU is slow enough that a live demo of several questions in a row
  takes a few minutes; this is a direct, disclosed trade-off of running a local model
  within the hardware constraints of this machine rather than a hosted API.
- Retrieval is plain cosine similarity with no reranking step. It works well at this corpus
  size (a few dozen chunks) but would need a proper vector index and reranker to scale.
- The revision cap is fixed at one retry. A more sophisticated system might vary this based
  on how close the first attempt came to passing verification.

## AI tool usage disclosure

This project was built using Cursor (AI coding assistant) and Claude for planning,
scaffolding, and iterative debugging. Every file was reviewed, tested, and in several
cases corrected after finding real bugs during testing — including an off-by-one error in
the verification retry counter that silently skipped the intended single retry, a dropped
`warnings` field that never reached the final schema output despite being computed
correctly upstream, and a triage misclassification on escalation-worthy questions that
required adding a new deterministic rule. These are documented in the commit history.
