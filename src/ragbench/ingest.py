"""Read raw documents (PDF, HTML, TXT, MD) and write chunks to JSONL.

Usage:
    python -m ragbench.ingest --raw data/raw --out data/processed/chunks.jsonl
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .chunking import chunk_document, save_chunks, split_sections
from .config import get_settings


def read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text("\n")


READERS = {
    ".pdf": read_pdf,
    ".html": read_html,
    ".htm": read_html,
    ".txt": lambda p: p.read_text(encoding="utf-8", errors="ignore"),
    ".md": lambda p: p.read_text(encoding="utf-8", errors="ignore"),
}


def clean(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"-\n(?=\w)", "", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ingest(raw_dir: Path, out_path: Path, max_words: int, overlap: int) -> int:
    files = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in READERS)
    if not files:
        raise SystemExit(f"No supported files found in {raw_dir} (expected {sorted(READERS)})")

    all_chunks = []
    for path in files:
        text = clean(READERS[path.suffix.lower()](path))
        chunks = chunk_document(path.stem.lower(), text, max_words=max_words, overlap=overlap)
        names = [name for name, _ in split_sections(text)]
        numbers = [int(n.split()[1]) for n in names if n.startswith("Article")]
        annexes = sum(n.startswith("Annex") for n in names)
        span = f"Articles {numbers[0]}-{numbers[-1]}" if numbers else "no article found"
        print(f"{path.name}: {len(text.split()):>7} words -> {len(chunks):>5} chunks | {span}, {annexes} annexes")
        # Headings are detected in sequence; a hole means a heading was missed by the PDF extraction,
        # so its text is attached to the previous article.
        missing = sorted(set(range(numbers[0], numbers[-1] + 1)) - set(numbers)) if numbers else []
        if missing:
            print(f"  WARNING articles not detected (merged into the previous one): {missing}")
        all_chunks.extend(chunks)

    save_chunks(all_chunks, out_path)
    print(f"Wrote {len(all_chunks)} chunks to {out_path}")
    return len(all_chunks)


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=s.raw_dir)
    ap.add_argument("--out", type=Path, default=s.chunks_path)
    ap.add_argument("--max-words", type=int, default=s.max_words)
    ap.add_argument("--overlap", type=int, default=s.overlap_words)
    args = ap.parse_args()
    ingest(args.raw, args.out, args.max_words, args.overlap)


if __name__ == "__main__":
    main()
