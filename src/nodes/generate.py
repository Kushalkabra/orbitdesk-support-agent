"""
Generation node: draft an answer from retrieved passages using the local LLM.
"""

import re
import time

import torch

from src.models import get_generator
from src.state import AgentState, RetrievedChunk

CITATION_PATTERN = re.compile(r"\[([^\]]+)\]")


INFERRED_CITATION_THRESHOLD = 0.35


def _format_passages(retrieved: list[RetrievedChunk]) -> str:
    if not retrieved:
        return "(No passages retrieved.)"
    formatted = []
    for chunk in retrieved:
        source_id = chunk["source_id"]
        sec_title = chunk.get("section_title", "").strip() if chunk.get("section_title") else ""
        if sec_title:
            formatted.append(f"[{source_id} {sec_title}] {chunk['passage']}")
        else:
            formatted.append(f"[{source_id}] {chunk['passage']}")
    return "\n\n".join(formatted)


def _build_messages(state: AgentState) -> list[dict[str, str]]:
    passages = _format_passages(state.get("retrieved", []))
    instructions = (
        "You are an OrbitDesk support assistant.\n"
        "- Only use the provided passages below.\n"
        "- Never include URLs, links, or web addresses in the answer, since none exist in the "
        "source material — if you want to reference something, use the bracketed source label "
        "only (e.g. [KB-003 Changing the Timezone] or [KB-003]), never a hyperlink.\n"
        "- If the passages do not answer the question, say plainly that the supplied "
        "documentation does not contain enough information rather than guessing.\n"
        "- CRITICAL MANDATORY REQUIREMENT: Every factual claim in your answer MUST end with a "
        "bracketed source reference matching the passage label (e.g. [KB-002] or [KB-003 Changing the Timezone]). "
        "Example of a correctly cited sentence: \"Owners and Admins can update workspace settings from Settings > Workspace [KB-003].\" "
        "An answer with no bracketed citation anywhere will be rejected."
    )

    user_content = f"Passages:\n{passages}\n\nQuestion: {state['question']}"

    if verification_reason := state.get("verification_reason"):
        user_content += (
            f"\n\nYour previous answer failed verification for this reason: "
            f"{verification_reason}\n"
            "Please revise your answer to fix that specific problem."
        )

    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_content},
    ]


def parse_citations(text: str, retrieved: list[RetrievedChunk]) -> tuple[list[dict[str, str]], list[str]]:
    """Extract citations that match retrieved chunks by source_id, preserving order.

    If no explicit citations are found, falls back to attaching the single highest-scoring
    retrieved chunk if its similarity score meets INFERRED_CITATION_THRESHOLD.
    """
    known_ids = {chunk["source_id"] for chunk in retrieved}
    chunks_by_id: dict[str, list[RetrievedChunk]] = {}
    for chunk in retrieved:
        chunks_by_id.setdefault(chunk["source_id"], []).append(chunk)

    parsed: list[dict[str, str]] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for match in CITATION_PATTERN.finditer(text):
        bracket_content = match.group(1).strip()
        matched_id = None
        matched_chunk = None

        for source_id in known_ids:
            if re.search(rf"\b{re.escape(source_id)}\b", bracket_content):
                matched_id = source_id
                chunks = chunks_by_id[source_id]
                matched_chunk = chunks[0]
                for chunk in chunks:
                    sec = chunk.get("section_title", "")
                    if sec and sec.lower() in bracket_content.lower():
                        matched_chunk = chunk
                        break
                break

        if matched_id and matched_id not in seen:
            seen.add(matched_id)
            sec_title = matched_chunk.get("section_title", "")
            if sec_title:
                print(f"Parsed citation: source_id={matched_id!r}, section_title={sec_title!r}")
            else:
                print(f"Parsed citation: source_id={matched_id!r}")
            parsed.append({
                "source_id": matched_id,
                "passage": matched_chunk["passage"],
            })

    for chunk in retrieved:
        source_id = chunk["source_id"]
        if source_id not in seen and re.search(rf"\b{re.escape(source_id)}\b", text):
            seen.add(source_id)
            sec_title = chunk.get("section_title", "")
            if sec_title:
                print(f"Parsed bare citation: source_id={source_id!r}, section_title={sec_title!r}")
            else:
                print(f"Parsed bare citation: source_id={source_id!r}")
            parsed.append({
                "source_id": source_id,
                "passage": chunk["passage"],
            })

    # Fallback: if zero citations found in raw text, use top retrieval match if score >= threshold
    if not parsed and retrieved:
        top_chunk = max(retrieved, key=lambda c: c.get("score", 0.0))
        if top_chunk.get("score", 0.0) >= INFERRED_CITATION_THRESHOLD:
            parsed.append({
                "source_id": top_chunk["source_id"],
                "passage": top_chunk["passage"],
            })
            warning_msg = "citation inferred from top retrieval match, not explicitly cited by model."
            warnings.append(warning_msg)
            sec_title = top_chunk.get("section_title", "")
            print(
                f"[PARSE_CITATIONS] Fallback: inferred citation from top match "
                f"source_id={top_chunk['source_id']!r}, section_title={sec_title!r} "
                f"(score={top_chunk.get('score', 0.0):.3f})"
            )

    return parsed, warnings


def run_generation(state: AgentState) -> dict:
    model, tokenizer = get_generator()
    messages = _build_messages(state)

    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = "\n\n".join(f"{message['role']}: {message['content']}" for message in messages)
        prompt += "\nassistant:"

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    prompt_length = input_ids.shape[1]
    generate_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": 300,
        "do_sample": False,
    }
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(**generate_kwargs)
    latency = time.time() - start
    print(f"Generation latency: {latency:.2f}s")

    new_token_ids = output_ids[0, prompt_length:]
    draft_answer = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    retrieved = state.get("retrieved", [])
    parsed_sources, warnings = parse_citations(draft_answer, retrieved)

    return {
        "draft_answer": draft_answer,
        "parsed_sources": parsed_sources,
        "warnings": warnings,
    }
