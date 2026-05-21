"""
Split long PDF text into smaller chunks for embedding and retrieval.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Break documents into overlapping chunks.

    Smaller chunks improve retrieval accuracy; overlap keeps
    sentences that span chunk boundaries from being cut off.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_documents(documents)
