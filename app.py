"""
College RAG Assistant — Streamlit app entry point.

Upload college PDFs, build a FAISS vector index, and chat with Gemini
using only content from your documents.
"""

import os
from pathlib import Path

# Reduce CPU/RAM pressure before scientific libraries load
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv

load_dotenv()

from utils.paths import FAISS_INDEX_DIR, setup_data_paths
from utils.ssl_setup import configure_ssl

setup_data_paths()
configure_ssl()

import streamlit as st

from utils.pdf_loader import load_pdfs_from_uploads
from utils.rag_chain import format_source_chunks, generate_answer
from utils.text_splitter import split_documents
from utils.vectorstore import (
    load_vectorstore,
    merge_vectorstores,
    save_vectorstore,
    search_similar_chunks,
)

# Always use TF-IDF locally (no PyTorch / sentence-transformers)
os.environ["EMBEDDING_MODE"] = "tfidf"

# ----- Page setup -----
st.set_page_config(
    page_title="College RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----- Custom styling for a clean, modern look -----
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #5a6b7d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .stChatMessage {
        border-radius: 12px;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize Streamlit session variables on first run."""
    defaults = {
        "vectorstore": None,
        "chat_history": [],  # List of {"role": "user"|"assistant", "content": str, "sources": list}
        "uploaded_file_names": [],
        "index_ready": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def try_load_saved_index():
    """Load FAISS index from disk if the app was used before."""
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = load_vectorstore()
        if st.session_state.vectorstore is not None:
            st.session_state.index_ready = True


def process_uploaded_pdfs(uploaded_files):
    """Full pipeline: PDF -> text -> chunks -> embeddings -> FAISS."""
    with st.spinner("Reading PDFs and building knowledge base..."):
        try:
            documents = load_pdfs_from_uploads(uploaded_files)
            if not documents:
                st.error("No text could be extracted from the uploaded PDFs.")
                return

            chunks = split_documents(documents)
            st.session_state.vectorstore = merge_vectorstores(
                st.session_state.vectorstore, chunks
            )
            save_vectorstore(st.session_state.vectorstore)

            new_names = [f.name for f in uploaded_files]
            st.session_state.uploaded_file_names = list(
                set(st.session_state.uploaded_file_names + new_names)
            )
            st.session_state.index_ready = True
            st.success(
                f"Indexed {len(chunks)} text chunks from {len(uploaded_files)} PDF(s)."
            )
        except Exception as exc:
            st.error("Could not build the vector database.")
            st.exception(exc)


def render_chat_history():
    """Display previous messages in the chat window."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📎 Source chunks used"):
                    for src in message["sources"]:
                        st.markdown(
                            f"**Chunk {src['chunk_id']}** — "
                            f"`{src['file']}` (page {src['page']})"
                        )
                        st.text(src["content"][:500] + ("..." if len(src["content"]) > 500 else ""))


def handle_user_question(question: str):
    """Retrieve context, ask Gemini, and append to chat history."""
    if not st.session_state.index_ready or st.session_state.vectorstore is None:
        st.warning("Please upload and process college PDFs first.")
        return

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Add `GOOGLE_API_KEY` to your `.env` file to use chat.")
        return

    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("Searching documents and generating answer..."):
        chunks = search_similar_chunks(st.session_state.vectorstore, question, k=4)

        try:
            answer = generate_answer(question, chunks)
        except Exception as e:
            err = str(e)
            if "SSL" in err or "CERTIFICATE" in err.upper():
                answer = (
                    "SSL certificate error connecting to Gemini. "
                    "Add `SSL_VERIFY=false` to your `.env` file (common on college Wi‑Fi), "
                    "then refresh the page and try again."
                )
            elif "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                answer = (
                    "Gemini API quota exceeded. Wait a minute and try again, "
                    "or create a new API key at https://aistudio.google.com/apikey"
                )
            else:
                answer = f"Could not get an answer from Gemini: {err}"
            chunks = []

        sources = format_source_chunks(chunks)
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )


# ----- Main app -----
def main():
    init_session_state()
    try_load_saved_index()

    # Sidebar: uploads and settings
    with st.sidebar:
        st.markdown("### 📚 Knowledge Base")
        st.markdown(
            "Upload college handbooks, syllabi, or policy PDFs. "
            "Answers are generated **only** from these files."
        )

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="You can select multiple PDFs at once.",
        )

        if st.button("Process PDFs", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("Please select at least one PDF.")
            else:
                process_uploaded_pdfs(uploaded_files)

        if st.session_state.uploaded_file_names:
            st.markdown("**Indexed files:**")
            for name in st.session_state.uploaded_file_names:
                st.caption(f"• {name}")

        if st.session_state.index_ready:
            st.success("Vector database is ready.")

        st.divider()
        st.markdown("### ⚙️ Settings")
        api_key_set = bool(os.getenv("GOOGLE_API_KEY"))
        if api_key_set:
            st.caption("✅ Gemini API key loaded")
        else:
            st.error("Set `GOOGLE_API_KEY` in your `.env` file.")

        st.caption(f"Data folder: `{FAISS_INDEX_DIR}`")
        st.caption("Embeddings: TF-IDF (local, no PyTorch)")

        if st.button("Clear chat history", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        if st.button("Reset knowledge base", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.uploaded_file_names = []
            st.session_state.index_ready = False
            st.session_state.chat_history = []
            if FAISS_INDEX_DIR.exists():
                for f in FAISS_INDEX_DIR.glob("*"):
                    f.unlink()
            st.rerun()

    # Main area
    st.markdown('<p class="main-header">🎓 College RAG Assistant</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ask questions about your college — answers come only from your uploaded PDFs.</p>',
        unsafe_allow_html=True,
    )

    if not os.getenv("GOOGLE_API_KEY"):
        st.info(
            "Copy `.env.example` to `.env` and add your [Gemini API key](https://aistudio.google.com/apikey)."
        )

    render_chat_history()

    # Chat input at the bottom
    if prompt := st.chat_input("Ask a question about your college documents..."):
        handle_user_question(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
