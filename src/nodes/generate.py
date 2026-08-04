"""
Generation node: draft an answer from retrieved passages using the local LLM.
"""

import re
import time

import torch

from src.models import get_generator
from src.state import AgentState, RetrievedChunk

CITATION_PATTERN = re.compile(r"\[([^\]]+)\]")


def _format_passages(retrieved: list[RetrievedChunk]) -> str:
    if not retrieved:
        return "(No passages retrieved.)"
    return "\n\n".join(f"[{chunk['source_id']}] {chunk['passage']}" for chunk in retrieved)


def _build_messages(state: AgentState) -> list[dict[str, str]]:
    passages = _format_passages(state.get("retrieved", []))
    instructions = (
        "You are an OrbitDesk support assistant.\n"
        "- Only use the provided passages below.\n"
        "- Cite the source_id(s) you use inline or on a clearly separated line, "
        "using the [source_id] format shown with each passage.\n"
        "- If the passages do not answer the question, say plainly that the supplied "
        "documentation does not contain enough information rather than guessing."
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


def parse_citations(text: str, retrieved: list[RetrievedChunk]) -> list[dict[str, str]]:
    """Extract [source_id] citations that match retrieved chunks, preserving order."""
    known_ids = {chunk["source_id"] for chunk in retrieved}
    passages_by_id = {chunk["source_id"]: chunk["passage"] for chunk in retrieved}

    parsed: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in CITATION_PATTERN.finditer(text):
        source_id = match.group(1)
        if source_id in known_ids and source_id not in seen:
            seen.add(source_id)
            parsed.append({"source_id": source_id, "passage": passages_by_id[source_id]})
    return parsed


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
    parsed_sources = parse_citations(draft_answer, retrieved)

    return {
        "draft_answer": draft_answer,
        "parsed_sources": parsed_sources,
    }
