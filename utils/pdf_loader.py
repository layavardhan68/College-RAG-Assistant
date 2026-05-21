"""
Extract plain text from uploaded college PDF files.
"""

from io import BytesIO

from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdfs_from_uploads(uploaded_files) -> list[Document]:
    """
    Read every uploaded PDF and return LangChain Document objects.

    Each document includes metadata (source filename, page number)
    so we can show which PDF/page an answer came from.
    """
    all_documents: list[Document] = []

    for uploaded_file in uploaded_files:
        # Read PDF bytes directly from Streamlit's upload widget
        pdf_bytes = uploaded_file.read()
        reader = PdfReader(BytesIO(pdf_bytes))

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue

            all_documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source_file": uploaded_file.name,
                        "page": page_num + 1,  # 1-based page numbers for users
                    },
                )
            )

    return all_documents
