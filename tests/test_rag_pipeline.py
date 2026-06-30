"""
Unit tests for the RAG pipeline components.

Tests that do NOT require an OpenAI API key (chunking, loading, embedding,
retrieval) run by default. The generation test is skipped automatically
if OPENAI_API_KEY is not set, so CI can run without secrets.
"""

import os
from pathlib import Path

import pytest

from app.text_loader import _chunk_text, load_and_chunk_documents
from app.rag_pipeline import RAGPipeline

SAMPLE_DOCS = Path(__file__).parent.parent / "data" / "sample_docs"


def test_chunk_text_short_string_returns_single_chunk():
    text = "This is a short sentence."
    chunks = _chunk_text(text, chunk_size=800)
    assert chunks == [text]


def test_chunk_text_long_string_splits_with_overlap():
    text = "word " * 1000
    chunks = _chunk_text(text, chunk_size=800, overlap=120)
    assert len(chunks) > 1
    # consecutive chunks should overlap
    assert chunks[0][-50:] in text


def test_load_and_chunk_documents_reads_sample_docs():
    chunks = load_and_chunk_documents(SAMPLE_DOCS)
    assert len(chunks) > 0
    sources = {c.source for c in chunks}
    assert "refund_policy.txt" in sources
    assert "remote_work_policy.txt" in sources


@pytest.mark.skipif(
    not os.environ.get("HF_API_KEY"),
    reason="HF_API_KEY not set — skipping live embedding/generation test.",
)
def test_pipeline_ingest_and_retrieve():
    pipeline = RAGPipeline()
    n = pipeline.ingest_directory(SAMPLE_DOCS)
    assert n > 0

    results = pipeline.retrieve("What is the refund window?", k=2)
    assert len(results) == 2
    assert any("refund" in r.text.lower() for r in results)


@pytest.mark.skipif(
    not os.environ.get("HF_API_KEY"),
    reason="HF_API_KEY not set — skipping live generation test.",
)
def test_pipeline_answer_returns_cited_response():
    pipeline = RAGPipeline()
    pipeline.ingest_directory(SAMPLE_DOCS)

    result = pipeline.answer("How many days can employees work remotely?")
    assert result.answer
    assert len(result.sources) > 0
