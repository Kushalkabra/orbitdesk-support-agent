import json
import re
from pathlib import Path

from src.models import get_embedder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KB_DIR = DATA_DIR / "knowledge_base"
CASES_FILE = DATA_DIR / "resolved_cases.json"


def _parse_document_id(text: str) -> str:
    match = re.search(r"^document_id:\s*(\S+)", text, re.MULTILINE)
    if not match:
        raise ValueError("Markdown file missing document_id in frontmatter")
    return match.group(1)


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4 :].lstrip("\n")
    return text


def _split_by_headers(text: str) -> list[dict]:
    sections: list[dict] = []
    current_lines: list[str] = []
    current_title: str = ""

    for line in _strip_frontmatter(text).splitlines():
        if line.startswith("## "):
            if current_lines:
                passage = "\n".join(current_lines).strip()
                if passage:
                    sections.append({"section_title": current_title, "passage": passage})
            current_lines = [line]
            current_title = line[3:].strip()
        elif line.startswith("# ") and not current_title:
            current_lines.append(line)
            current_title = line[2:].strip()
        else:
            current_lines.append(line)

    if current_lines:
        passage = "\n".join(current_lines).strip()
        if passage:
            sections.append({"section_title": current_title, "passage": passage})

    return sections


def _load_kb_chunks() -> list[dict]:
    chunks: list[dict] = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        source_id = _parse_document_id(text)
        for section in _split_by_headers(text):
            chunks.append({
                "source_id": source_id,
                "section_title": section["section_title"],
                "passage": section["passage"],
            })
    return chunks


def _flatten_case(case: dict) -> str:
    lines = [case["title"], *case.get("symptoms", []), *case.get("resolution", [])]
    if reason := case.get("superseded_reason"):
        lines.append(reason)
    return "\n".join(lines)


def _load_case_chunks() -> list[dict]:
    data = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    return [
        {
            "source_id": case["case_id"],
            "section_title": case.get("title", ""),
            "passage": _flatten_case(case),
        }
        for case in data["cases"]
    ]


def build_index() -> list[dict]:
    chunks = _load_kb_chunks() + _load_case_chunks()
    passages = [chunk["passage"] for chunk in chunks]
    embeddings = get_embedder().encode(passages, convert_to_numpy=True)
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    return chunks
