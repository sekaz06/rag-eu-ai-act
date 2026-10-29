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

# Candidate headings: "Article 5" or "ARTICLE 50" at the start of a line. Cross-references that
# wrap onto a new line ("Article 6 of Regulation ...") also match, so candidates are filtered
# in `find_article_headings`. "Article 5(1)" is excluded right away.
ARTICLE_RE = re.compile(r"^[ \t]*(article\s+(\d+))\b(?!\s*\()", re.IGNORECASE | re.MULTILINE)
# Annexes of EU regulations: "ANNEX III" alone on its line, after the last article.
ANNEX_RE = re.compile(r"^[ \t]*(ANNEX\s+[IVXLC]+)[ \t]*$", re.MULTILINE)
MAX_GAP = 3  # tolerate a few undetected headings before giving up on the sequence


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


def _is_standalone(text: str, m: re.Match) -> bool:
    """True when the heading is alone on its line, as real article titles are in EUR-Lex texts."""
    end = text.find("\n", m.end())
    return not text[m.end() : end if end != -1 else len(text)].strip()


def find_article_headings(text: str) -> list[tuple[int, str]]:
    """Return (position, name) of the real article headings.

    Articles of a regulation appear in increasing order, so we walk through the candidates and
    only accept the next expected number (tolerating up to MAX_GAP missed headings). Among the
    candidates for that number, one alone on its line is preferred: a candidate followed by
    text on the same line is usually a cross-reference.
    """
    cands = [
        (m.start(), int(m.group(2)), _normalize_section(m.group(1)), _is_standalone(text, m))
        for m in ARTICLE_RE.finditer(text)
    ]
    if not cands:
        return []

    headings: list[tuple[int, str]] = []
    pos = 0
    # Start at Article 1, or (for an excerpt) at the first heading alone on its line.
    first = next((c for c in cands if c[3]), cands[0])
    expected = 1 if any(c[1] == 1 for c in cands) else first[1]
    while True:
        chosen = None
        for num in range(expected, expected + MAX_GAP + 1):
            pool = [c for c in cands if c[0] >= pos and c[1] == num]
            if pool:
                chosen = next((c for c in pool if c[3]), pool[0])
                break
        if chosen is None:
            # Larger hole: resume at the next heading alone on its line with a higher number.
            chosen = next((c for c in cands if c[0] >= pos and c[1] > expected and c[3]), None)
        if chosen is None:
            return headings
        headings.append((chosen[0], chosen[2]))
        pos, expected = chosen[0] + 1, chosen[1] + 1


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split a document into (section_name, section_text) pairs: preamble, articles, annexes."""
    headings = find_article_headings(text)
    if not headings:
        return [("full", text)]

    last_article = headings[-1][0]
    headings += [
        (m.start(), "Annex " + m.group(1).split()[1]) for m in ANNEX_RE.finditer(text) if m.start() > last_article
    ]

    sections: list[tuple[str, str]] = []
    head = text[: headings[0][0]]
    if head.strip():
        sections.append(("preambule", head))
    for i, (start, name) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        sections.append((name, text[start:end]))
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
