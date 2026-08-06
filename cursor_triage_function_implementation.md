# Triage function implementation
_Exported on 8/6/2026 at 16:54:58 GMT+5:30 from Cursor (3.14.7)_

---

**User**

Write src/nodes/triage.py, a node function run_triage(state) matching the AgentState TypedDict in src/state.py. It must return a partial dict with classification and triage_reason keys (and clarification_question if relevant).

Two stages:

Deterministic keyword rules, checked first, before any model call:
If the question mentions refund, billing, subscription cancellation, legal advice, or contains phrasing like "ignore the documentation" / "ignore your instructions" / "ignore the supplied material" → classification = out_of_scope. This is a prompt-injection defense, not just a topic filter, so match on intent-to-override phrasing, not just topic words.
If the question is extremely short/vague on its own — e.g. it only says something like "it's not working" or "sync is broken" with no object, error code, or specific symptom mentioned — classification = requires_clarification, and set clarification_question to something asking for the missing specifics (workspace ID, affected object, error code — see KB-006's "Troubleshooting" section for the exact fields it wants).
If neither rule matches, call get_classifier() from src/models.py with candidate labels ["answerable", "requires_clarification", "requires_escalation", "out_of_scope"] and take the top-scoring label.

Set triage_reason to a short human-readable explanation either way (which rule matched, or the classifier's top label and score). Keep the keyword lists as module-level constants near the top of the file, not inline in the function.

---

**Cursor**

I'll read the state schema, classifier API, and existing node patterns first so `run_triage` matches the project.


Creating `src/nodes/triage.py` with deterministic rules first, then classifier fallback.


Running a quick test of the deterministic rules without loading the classifier model.


`src/nodes/triage.py` is in place with `run_triage(state)` returning a partial `AgentState` update.

### Flow

1. **Out-of-scope (deterministic, first)** — keyword match on `refund`, `billing`, `subscription cancellation`, `legal advice`, etc., or prompt-injection patterns like `ignore the documentation`, `ignore your instructions`, `ignore the supplied material` (plus close variants).
2. **Requires clarification (deterministic)** — very short questions (≤12 words) or vague phrasing (`it's not working`, `sync is broken`, `sync is not working`, …) when there is no concrete object, error code, or symptom.
3. **Classifier fallback** — `get_classifier()` with labels `["answerable", "requires_clarification", "requires_escalation", "out_of_scope"]`; top label wins.

### Return shape

- Always: `classification`, `triage_reason`
- When clarification is needed: `clarification_question` (aligned with KB-006 Troubleshooting fields — workspace ID, connection name/ID, state, last refresh, error code, manual vs scheduled)

Keyword lists and regex patterns live as module-level constants at the top of the file.

Verified against sample questions:
- **Q-005** → `out_of_scope` (refund + injection)
- **Q-003** → `requires_clarification` with KB-006 clarification question
- **Q-004** → not flagged vague (`render_failed` counts as specificity)

---

**User**

Write src/nodes/retrieve.py, a node function run_retrieval(state) matching AgentState. Use build_index() from src/ingest.py (call it once at module load and cache the result — don't rebuild the index on every call) and get_embedder() from src/models.py. Embed state["question"], compute cosine similarity against every chunk's stored embedding using numpy (no external similarity library needed), sort descending, and return the top 4 as retrieved (list of {"source_id", "passage", "score"} matching the RetrievedChunk shape in state.py), plus top_score set to the highest score in that list. Do not filter out low scores here — that decision belongs to the verification node, not retrieval. Keep the cosine similarity as a small standalone function so it's testable on its own.

---

**Cursor**

Reviewing the ingest index format and AgentState shape so retrieval matches the project.


`src/nodes/retrieve.py` is in place.

### Structure

- **`cosine_similarity(a, b)`** — standalone numpy helper (returns `0.0` on zero-norm vectors)
- **`_INDEX = build_index()`** — built once at import, not on every call
- **`run_retrieval(state)`** — embeds `state["question"]`, scores all chunks, returns top 4

### Return value

```python
{
    "retrieved": [
        {"source_id": "...", "passage": "...", "score": 0.54},
        ...
    ],  # top 4, sorted descending
    "top_score": 0.54,  # highest score in that list
}
```

No score thresholding — low-scoring chunks are still returned for the verification node to handle.

Verified: cosine similarity behaves correctly on identical, orthogonal, and zero vectors; retrieval returns 4 `RetrievedChunk`-shaped dicts in descending score order (top hit for a sync question was `KB-006`).

---

**User**

Write src/schema.py using Pydantic v2. Create a model SourceRef with fields source_id: str and passage: str (min length 1), and a model SupportResponse with fields matching the provided output_schema.json: classification (Literal of the five values), answer: str, sources: list[SourceRef], confidence: float (ge=0, le=1), requires_human: bool, reason: str, clarification_question: str | None = None, warnings: list[str] = []. Set model_config = {"extra": "forbid"} to mirror additionalProperties: false in the original schema. No other logic in this file — just the models.

---

**Cursor**

Creating `src/schema.py` from your field spec; `output_schema.json` isn't in the repo.


`src/schema.py` defines two Pydantic v2 models:

- **`SourceRef`** — `source_id: str`, `passage: str` with `min_length=1`
- **`SupportResponse`** — all fields from the output schema:
  - `classification` — `Literal` of the five values (`answerable`, `requires_clarification`, `requires_escalation`, `out_of_scope`, `safe_failure`)
  - `answer`, `sources`, `confidence` (0–1), `requires_human`, `reason`
  - `clarification_question: str | None = None`
  - `warnings: list[str] = []`

Both models use `model_config = {"extra": "forbid"}` to match `additionalProperties: false`. No other logic in the file.

---

**User**

Write src/nodes/generate.py, a node function run_generation(state) matching AgentState. Build a prompt for the local LLM that:

Lists each retrieved chunk as [source_id] passage text.
Instructs the model explicitly: only use the provided passages, cite the source_id(s) used inline or in a clearly separated line, and say plainly if the passages don't answer the question rather than guessing.
If state.get("verification_reason") is set (meaning this is a revision after a failed verification), append that failure reason to the prompt and ask the model to fix that specific problem.

Use get_generator() from src/models.py (returns tokenizer + model), apply the tokenizer's chat template if available, generate with max_new_tokens=300, do_sample=False for reproducibility, decode only the newly generated tokens (not the echoed prompt). Parse out any [source_id] style citations the model included via a simple regex against the known source_ids in state["retrieved"], and store them separately as parsed_sources alongside the raw text in draft_answer. Return {"draft_answer": ..., "parsed_sources": [...]}. Time the generation call and print the latency, since I need to report it in my README.

---

**Cursor**

Reviewing state, models, and graph wiring so generation matches the existing node pattern.


Implementing `generate.py` with prompt building, chat-template generation, and citation parsing.


`src/nodes/generate.py` is in place with `run_generation(state)`.

### Prompt

- Each retrieved chunk is formatted as `[source_id] passage text`.
- System instructions require using only those passages, citing `[source_id]` inline or on a separate line, and stating plainly when the passages do not answer the question.
- On revision (`state.get("verification_reason")`), the failure reason is appended with a request to fix that specific issue.

### Generation

- Uses `get_generator()` (tokenizer + model).
- Applies the tokenizer chat template when available.
- Generates with `max_new_tokens=300`, `do_sample=False`.
- Decodes only tokens after the prompt (no echo).
- Prints `Generation latency: X.XXs`.

### Citation parsing

- `parse_citations(text, retrieved)` finds `[source_id]` markers via regex and keeps only IDs present in `state["retrieved"]`.
- Returns:

```python
{
    "draft_answer": "<raw model text>",
    "parsed_sources": [{"source_id": "...", "passage": "..."}, ...],
}
```

---

**User**

Write src/nodes/verify.py, a node function run_verification(state) matching AgentState. It should check, in order:

parsed_sources from the generate node is non-empty and every source_id in it actually exists in state["retrieved"]. If not, fail with reason "no valid source citations".
A lexical support check: for each sentence in draft_answer, check whether a reasonable fraction of its significant words (lowercased, stopwords removed) also appear in the retrieved passages combined. If most sentences have very low overlap, fail with reason "answer not grounded in retrieved evidence". Keep this heuristic simple and explicit — don't pull in a new NLI model here, just note in a comment that a cross-encoder or NLI entailment check would be a stronger version of this.
If both checks pass, build a SupportResponse (from src/schema.py) using state["classification"], draft_answer as answer, parsed_sources mapped into SourceRef objects, a confidence derived from state["top_score"], requires_human=False unless classification is requires_escalation, and reason summarizing why this passed. If the Pydantic validation itself throws, treat that as a failure too with reason "schema validation failed".

On success return {"verification_passed": True, "draft_answer_as_schema": response.model_dump()}. On failure return {"verification_passed": False, "verification_reason": ..., "revision_count": state.get("revision_count", 0) + 1}.

---

**Cursor**

Reviewing generate output, schema, and state so verification aligns with the graph.


Implementing verification with explicit citation and lexical grounding checks.


`src/nodes/verify.py` implements `run_verification(state)` with three checks in order:

### 1. Citation check
- `parsed_sources` must be non-empty
- Every `source_id` must exist in `state["retrieved"]`
- Otherwise fails with `"no valid source citations"`

### 2. Lexical grounding check
- Splits `draft_answer` into sentences (citation markers stripped first)
- For each sentence, computes overlap of significant words (lowercased, stopwords removed) against combined retrieved passages
- Fails with `"answer not grounded in retrieved evidence"` when more than half of scorable sentences fall below the 15% overlap threshold
- Includes a comment noting a cross-encoder/NLI check would be stronger

### 3. Schema build
On pass, builds `SupportResponse` with:
- `classification` from state
- `answer` = `draft_answer`
- `sources` from `parsed_sources` as `SourceRef` objects
- `confidence` = `top_score` clamped to `[0, 1]`
- `requires_human` = `True` only for `requires_escalation`
- `reason` summarizing successful verification

Pydantic validation errors map to `"schema validation failed"`.

### Returns

**Success:**
```python
{"verification_passed": True, "draft_answer_as_schema": response.model_dump()}
```

**Failure:**
```python
{
    "verification_passed": False,
    "verification_reason": "...",
    "revision_count": state.get("revision_count", 0) + 1,
}
```

Helpers `check_citations`, `check_lexical_grounding`, `significant_words`, and `sentence_word_overlap` are exposed for testing.

---

**User**

Write scripts/run_cli.py. It should import build_graph from src.graph, accept a question either as a command-line argument or interactively via input() if none is given, run the compiled graph with an initial state {"question": question, "revision_count": 0, "node_trace": []}, then pretty-print two things: the final_response as formatted JSON, and a plain-English summary line showing node_trace so I can see which nodes executed. Also append the full run (question, node_trace, final_response, timestamp) as one line of JSON to logs/run_log.jsonl for the required execution logs. Add a --question-id optional flag so I can tag runs against the sample_questions.json ids when demoing.

---

**Cursor**

Checking project layout and graph wiring before adding the CLI script.


`scripts/run_cli.py` is ready.

### Usage

```bash
python scripts/run_cli.py "Your question here"
python scripts/run_cli.py --question-id Q-003 "Our data sync is not working."
python scripts/run_cli.py   # prompts: Question:
```

### Behavior

1. Builds the graph via `build_graph()` from `src.graph`
2. Runs with initial state: `{"question": ..., "revision_count": 0, "node_trace": []}`
3. Prints `final_response` as indented JSON
4. Prints a one-line trace summary, e.g. `Nodes executed: run_triage -> run_retrieval -> ...`
5. Appends one JSON line per run to `logs/run_log.jsonl`:

```json
{
  "timestamp": "2026-08-04T13:26:00+00:00",
  "question": "...",
  "question_id": "Q-001",
  "node_trace": ["run_triage", "..."],
  "final_response": { ... }
}
```

`question_id` is included only when `--question-id` is passed. The script adds the repo root to `sys.path` so it can be run as `python scripts/run_cli.py` from the project root.

---

**User**

Write tests/test_routing.py using pytest. Import build_graph from src.graph. Load the five questions from data/sample_questions.json. Write these assertions without checking any exact answer text, only node_trace and classification:

For Q-005 (the refund / "ignore the documentation" question), assert final_response["classification"] == "out_of_scope" and assert "generate" never appears in node_trace — the graph must never reach generation for this one.
For Q-003 (the vague "sync is not working" question), assert final_response["classification"] == "requires_clarification" and final_response["clarification_question"] is not None.
For Q-001, Q-002, and Q-004, assert node_trace contains "run_triage", "run_retrieval", "run_generation", and "run_verification" in that relative order, and assert final_response["sources"] is non-empty.
Add one test that manually constructs a state that fails verification twice (mock or monkeypatch run_verification to always return verification_passed=False) and asserts the graph terminates with classification == "safe_failure" rather than looping forever — this is the infinite-loop guard test.

Run each real question through the actual compiled graph (not mocks) except for the loop-guard test, since we want to prove the real models route correctly, not just the logic in isolation.

---

**Cursor**

Reviewing sample questions and graph wiring before writing routing tests.


Creating the routing tests and pytest config, then running them.


Investigating test failures: checking final responses and fixing the loop-guard mock naming.


Fixing citation parsing so bare `KB-xxx` mentions are recognized (the model often omits brackets), and fixing the loop-guard mock name.


LangGraph drops `parsed_sources` because it's missing from `AgentState`. Adding it to the state schema.


`tests/test_routing.py` and `pytest.ini` are in place. All six tests pass against the real graph (~5.5 min on CPU).

### Tests

| Test | Checks |
|------|--------|
| `test_q005_out_of_scope` | `classification == "out_of_scope"`, no `generate` / `run_generation` in trace |
| `test_q003_requires_clarification` | `classification == "requires_clarification"`, `clarification_question` set |
| `test_answerable_questions_run_full_pipeline[Q-001/Q-002/Q-004]` | Pipeline nodes in order; `sources` non-empty |
| `test_verification_loop_guard` | Mocked verify always fails → `safe_failure`, bounded verify/generate runs |

### Supporting fixes (needed for real-graph tests)

1. **`src/state.py`** — Added `parsed_sources` and `draft_answer_as_schema` to `AgentState`. LangGraph was dropping `parsed_sources` from generate output because it wasn't in the TypedDict, which caused verify to always fail citations.

2. **`src/nodes/generate.py`** — Citation parser also matches bare `KB-xxx` / `CASE-xxx` mentions when the model omits brackets.

Run with:

```bash
pytest tests/test_routing.py -v
```

---

**User**

generate a git ignore file looking at the project

---

**Cursor**

Scanning the project layout to tailor `.gitignore`.


