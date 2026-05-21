"""
Send retrieved context to Gemini and return an answer grounded in PDFs only.
"""

import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.ssl_setup import get_ssl_verify_setting


def _format_context(chunks: list[Document]) -> str:
    """Turn retrieved chunks into one context block for the LLM."""
    parts = []
    for i, doc in enumerate(chunks, start=1):
        source = doc.metadata.get("source_file") or doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        parts.append(
            f"[Chunk {i} | File: {source} | Page: {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def get_gemini_model() -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model using the API key from environment."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add it to a .env file or set it as an environment variable."
        )

    # gemini-2.5-flash has broader free-tier availability than 2.0-flash
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2,
        client_args={"verify": get_ssl_verify_setting()},
    )


# Prompt tells Gemini to answer ONLY from the provided college PDF context
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful College Assistant. Answer the student's question using ONLY
the context below from uploaded college PDF documents.

Rules:
- If the answer is not in the context, say: "I could not find that information in the uploaded college PDFs."
- Do not use outside knowledge or guess.
- Be clear, accurate, and concise.
- When helpful, mention which document or topic the information relates to.

Context from college PDFs:
{context}""",
        ),
        ("human", "{question}"),
    ]
)


def generate_answer(question: str, chunks: list[Document]) -> str:
    """
    Run the RAG pipeline: context + question -> Gemini -> answer string.
    """
    if not chunks:
        return "No relevant content was found in the uploaded PDFs. Please upload documents first."

    context = _format_context(chunks)
    llm = get_gemini_model()

    chain = RAG_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def format_source_chunks(chunks: list[Document]) -> list[dict]:
    """
    Prepare source chunk info for display in the Streamlit UI.
    """
    sources = []
    for i, doc in enumerate(chunks, start=1):
        sources.append(
            {
                "chunk_id": i,
                "file": doc.metadata.get("source_file") or doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "content": doc.page_content,
            }
        )
    return sources
