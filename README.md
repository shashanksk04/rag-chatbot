# 📚 Enterprise RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about
your private documents — with cited sources and zero hallucinated answers
outside the provided context.

Built to demonstrate a production-shaped RAG pipeline: document chunking,
dense embeddings, vector search, and grounded LLM generation, wrapped in a
Streamlit UI and containerized for deployment.

## How it works

```
 ┌─────────────┐     ┌────────────────┐     ┌─────────────┐     ┌──────────────┐
 │  Documents   │ --> │  Chunk + Embed  │ --> │ FAISS Index  │ --> │ Vector Search │
 │ (.txt/.pdf)  │     │ (MiniLM-L6-v2)  │     │ (cosine sim) │     │  top-k chunks │
 └─────────────┘     └────────────────┘     └─────────────┘     └──────┬───────┘
                                                                          │
                                                                          v
                                                          ┌───────────────────────────┐
                                                          │  GPT-4o-mini generation    │
                                                          │  (grounded, cited answer)  │
                                                          └───────────────────────────┘
```

1. **Ingest** — documents are loaded and split into overlapping ~800-character chunks.
2. **Embed** — each chunk is embedded using a HuggingFace `sentence-transformers` model (`all-MiniLM-L6-v2`).
3. **Index** — embeddings are stored in a FAISS `IndexFlatIP` index for fast cosine-similarity search.
4. **Retrieve** — the user's query is embedded and the top-k most similar chunks are retrieved.
5. **Generate** — retrieved chunks + the query are passed to HuggingFace (free)'s `Llama-3-8B-Instruct`, which is instructed to answer **only** from the provided context and cite sources.

## Quickstart

### 1. Local (Python)

```bash
git clone <this-repo>
cd rag-chatbot
pip install -r requirements.txt

cp .env.example .env   # add your HF_API_KEY

streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`, upload a few `.txt`/`.md`/`.pdf` files in the sidebar, click **Build Index**, and start asking questions.

### 2. Docker

```bash
export HF_API_KEY=sk-...
docker compose up --build
```

### 3. Command line (no UI)

```bash
python -m app.cli --data data/sample_docs --query "What is the refund policy?"
```

## Sample data

`data/sample_docs/` includes two example policy documents (refund policy,
remote work policy) so you can try the pipeline immediately without
uploading your own files.

Example query against the sample docs:

```bash
python -m app.cli --data data/sample_docs --query "How many days can I work remotely?"
```

## Running tests

```bash
pytest tests/ -v
```

Tests that require an HuggingFace (free) key (live embedding/generation) are
automatically skipped if `HF_API_KEY` is not set, so the test suite
runs cleanly in CI without secrets configured.

## Project structure

```
rag-chatbot/
├── app/
│   ├── rag_pipeline.py     # Core RAG logic (ingest, retrieve, generate)
│   ├── text_loader.py      # Document loading + chunking
│   ├── streamlit_app.py    # Web UI
│   └── cli.py               # Command-line interface
├── data/sample_docs/        # Example documents for demo/testing
├── tests/                   # Pytest unit tests
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml # GitHub Actions: test + lint + docker build
└── requirements.txt
```

## Design decisions

- **FAISS over a managed vector DB** — keeps the project self-contained and free to run; swapping in Pinecone/Weaviate would mean changing only `rag_pipeline.py`.
- **MiniLM embeddings** — small, fast, and good enough for demo-scale corpora (384 dimensions), runs on CPU with no GPU required.
- **Strict grounding prompt** — the system prompt explicitly forbids answering outside the retrieved context, which is the main lever for reducing hallucination in RAG systems.
- **CLI + Streamlit both included** — the CLI makes the pipeline easy to test/demo in a terminal or CI job; Streamlit gives a recruiter-friendly visual demo.

## Tech stack

`Python` · `LangChain-style RAG architecture` · `HuggingFace (free) API (Llama-3-8B-Instruct)` · `HuggingFace sentence-transformers` · `FAISS` · `Streamlit` · `Docker` · `GitHub Actions`

## License

MIT
