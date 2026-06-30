"""
Streamlit front-end for the RAG chatbot.

Run with:  streamlit run app/streamlit_app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

from app.rag_pipeline import RAGPipeline

st.set_page_config(page_title="RAG Document Chatbot", page_icon="📚", layout="wide")

st.title("📚 Enterprise RAG Chatbot")
st.caption("Upload documents, then ask questions. Answers are grounded in your files with cited sources.")

# ---------------------------------------------------------------------- #
# Session state
# ---------------------------------------------------------------------- #
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "history" not in st.session_state:
    st.session_state.history = []
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

# ---------------------------------------------------------------------- #
# Sidebar: API key + document upload
# ---------------------------------------------------------------------- #
with st.sidebar:
    st.header("Setup")

    api_key = st.text_input(
        "HuggingFace API Token",
        type="password",
        value=os.environ.get("HF_API_KEY", ""),
        help="Get a free token at huggingface.co/settings/tokens. Used only for this session, never stored.",
    )

    uploaded_files = st.file_uploader(
        "Upload documents (.txt, .md, .pdf)",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    if st.button("Build Index", type="primary", disabled=not (api_key and uploaded_files)):
        with st.spinner("Embedding documents and building FAISS index..."):
            tmp_dir = tempfile.mkdtemp()
            for f in uploaded_files:
                (Path(tmp_dir) / f.name).write_bytes(f.read())

            pipeline = RAGPipeline(hf_api_key=api_key)
            count = pipeline.ingest_directory(tmp_dir)

            st.session_state.pipeline = pipeline
            st.session_state.doc_count = count
            st.session_state.history = []

        st.success(f"Indexed {count} chunks from {len(uploaded_files)} file(s).")

    if st.session_state.pipeline:
        st.metric("Indexed chunks", st.session_state.doc_count)

# ---------------------------------------------------------------------- #
# Main chat area
# ---------------------------------------------------------------------- #
if not st.session_state.pipeline:
    st.info("⬅️ Add your OpenAI API key and upload at least one document to get started.")
else:
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander("Sources"):
                    for s in turn["sources"]:
                        st.markdown(f"**{s.source}** (score: {s.score:.3f})")
                        st.caption(s.text[:300] + "...")

    query = st.chat_input("Ask a question about your documents...")
    if query:
        st.session_state.history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant chunks and generating answer..."):
                result = st.session_state.pipeline.answer(query)
            st.markdown(result.answer)
            with st.expander("Sources"):
                for s in result.sources:
                    st.markdown(f"**{s.source}** (score: {s.score:.3f})")
                    st.caption(s.text[:300] + "...")

        st.session_state.history.append(
            {"role": "assistant", "content": result.answer, "sources": result.sources}
        )
