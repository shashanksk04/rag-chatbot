"""
Document loader and chunker.

Walks a directory, reads .txt/.md/.pdf files, and splits them into
overlapping chunks suitable for embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    source: str


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """Simple sliding-window chunker on raw characters."""
    text = " ".join(text.split())  # normalize whitespace
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def load_and_chunk_documents(directory: str | Path) -> List[Chunk]:
    directory = Path(directory)
    supported = {".txt", ".md", ".pdf"}
    chunks: List[Chunk] = []

    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in supported or not path.is_file():
            continue

        if path.suffix.lower() == ".pdf":
            raw_text = _read_pdf_file(path)
        else:
            raw_text = _read_text_file(path)

        for piece in _chunk_text(raw_text):
            if piece.strip():
                chunks.append(Chunk(text=piece, source=path.name))

    return chunks
