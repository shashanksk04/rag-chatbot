"""
RAG Pipeline — Core retrieval-augmented generation logic.

Ingests documents, builds a FAISS vector index using HuggingFace
sentence-transformer embeddings, and answers natural-language queries
with cited sources using the HuggingFace Inference API (free tier).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import faiss
import numpy as np
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

from app.text_loader import load_and_chunk_documents


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


@dataclass
class RAGAnswer:
    answer: str
    sources: List[RetrievedChunk] = field(default_factory=list)


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    1. Ingest:   chunk raw documents -> embed with sentence-transformers -> store in FAISS
    2. Retrieve: embed the user query -> nearest-neighbour search in FAISS
    3. Generate: pass retrieved chunks + query to an LLM, return a cited answer
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chat_model: str = "openai/gpt-oss-120b:groq",
        top_k: int = 4,
        hf_api_key: str | None = None,
    ) -> None:
        self.embedder = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # cosine sim via normalized vectors
        self.chunks: List[RetrievedChunk] = []
        self.chat_model = chat_model
        self.top_k = top_k

        api_key = hf_api_key or os.environ.get("HF_API_KEY") or os.environ.get("HUGGINGFACE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "HF_API_KEY not set. Get a free token at https://huggingface.co/settings/tokens "
                "then export it or pass hf_api_key=... explicitly."
            )
        self.client = InferenceClient(model=self.chat_model, token=api_key)

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest_directory(self, directory: str | Path) -> int:
        """Load every .txt/.md/.pdf file in `directory`, chunk it, and index it."""
        documents = load_and_chunk_documents(directory)
        if not documents:
            raise ValueError(f"No ingestible documents found in {directory}")

        texts = [d.text for d in documents]
        embeddings = self._embed(texts)
        self.index.add(embeddings)

        for doc in documents:
            self.chunks.append(RetrievedChunk(text=doc.text, source=doc.source, score=0.0))

        return len(documents)

    def _embed(self, texts: List[str]) -> np.ndarray:
        vectors = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(vectors)
        return vectors.astype("float32")

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, k: int | None = None) -> List[RetrievedChunk]:
        if self.index.ntotal == 0:
            raise RuntimeError("Index is empty. Call ingest_directory() first.")

        k = k or self.top_k
        query_vec = self._embed([query])
        scores, indices = self.index.search(query_vec, k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(RetrievedChunk(text=chunk.text, source=chunk.source, score=float(score)))
        return results

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def answer(self, query: str, k: int | None = None) -> RAGAnswer:
        retrieved = self.retrieve(query, k=k)
        context = "\n\n".join(
            f"[Source: {c.source}]\n{c.text}" for c in retrieved
        )

        system_prompt = (
            "You are a helpful assistant that answers ONLY using the provided context. "
            "If the answer is not contained in the context, say you don't have enough "
            "information. Always cite the source file name in brackets after each claim."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=512,
            temperature=0.2,
        )

        return RAGAnswer(
            answer=response.choices[0].message.content,
            sources=retrieved,
        )
