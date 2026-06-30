"""
Command-line entry point for the RAG pipeline — useful for quick testing
without spinning up Streamlit.

Usage:
    python -m app.cli --data data/sample_docs --query "What is the refund policy?"
"""

import argparse

from app.rag_pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a RAG pipeline from the command line.")
    parser.add_argument("--data", required=True, help="Directory of documents to ingest.")
    parser.add_argument("--query", required=True, help="Question to ask.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    pipeline = RAGPipeline(top_k=args.top_k)
    n = pipeline.ingest_directory(args.data)
    print(f"Indexed {n} chunks from '{args.data}'\n")

    result = pipeline.answer(args.query)

    print("ANSWER:")
    print(result.answer)
    print("\nSOURCES:")
    for s in result.sources:
        print(f"  - {s.source} (score: {s.score:.3f})")


if __name__ == "__main__":
    main()
