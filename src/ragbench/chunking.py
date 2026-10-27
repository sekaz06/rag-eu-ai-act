"""Structure-aware chunking.

Legal texts are split on their own structure ("Article N") first, then long
articles are cut into overlapping word windows. Every chunk keeps a
``section_key`` (``doc_id::Article N``) so that relevance labels in the
evaluation set stay valid when the chunking strategy changes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

# "Article 5", "Article 6 bis", "ARTICLE 50" at the start of a line
ARTICLE_RE = re.compile(r"^[ \t]*(article\s+\d+(?:\s*(?:bis|ter|quater))?)\b", re.IGNORECASE | re.MULTILINE)


@dataclass
class Chunk:
    id: str
    doc_id: str
    section: str
    text: str

    @property
    def section_key(self) -> str:
        return f"{self.doc_id}::{self.section}"


def _normalize_section(name: str) -> str:
    name = re.sub(r"\s+", " ", name.strip())
    return name[0].upper() + name[1:].lower()


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split a document into (section_name, section_text) pairs."""
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        return [("full", text)]

    sections: list[tuple[str, str]] = []
    head = text[: matches[0].start()]
    if head.strip():
        sections.append(("preambule", head))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((_normalize_section(m.group(1)), text[m.start() : end]))
    return sections


def word_windows(words: list[str], size: int, overlap: int) -> Iterator[list[str]]:
    if size <= overlap:
        raise ValueError("size must be greater than overlap")
    if len(words) <= size:
        yield words
        return
    step = size - overlap
    for start in range(0, len(words), step):
        yield words[start : start + size]
        if start + size >= len(words):
            break


def chunk_document(doc_id: str, text: str, max_words: int = 250, overlap: int = 50) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, body in split_sections(text):
        words = body.split()
        if not words:
            continue
        for window in word_windows(words, max_words, overlap):
            # Prefixing the location helps both BM25 and the embedding model.
            content = f"[{doc_id} — {section}]\n" + " ".join(window)
            chunks.append(Chunk(id=f"{doc_id}-{len(chunks):05d}", doc_id=doc_id, section=section, text=content))
    return chunks


def save_chunks(chunks: Iterable[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with Path(path).open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
