# College RAG Assistant

A beginner-friendly **Retrieval-Augmented Generation (RAG)** chatbot for college documents. Upload PDFs (handbooks, syllabi, policies), then ask questions and get answers grounded **only** in those files—powered by **FAISS**, **HuggingFace embeddings**, and **Google Gemini**.

## Features

- Upload multiple college PDFs
- Extract and chunk text automatically
- HuggingFace `all-MiniLM-L6-v2` embeddings stored in **FAISS**
- Chat interface with **history**
- Retrieves relevant chunks and sends them to **Gemini**
- Shows **source chunks** used for each answer
- Persists the vector index on disk between sessions

## Project structure

```
college-rag-assistant/
├── app.py                 # Streamlit UI and main flow
├── requirements.txt
├── .env.example
├── utils/
│   ├── pdf_loader.py      # PDF text extraction
│   ├── text_splitter.py   # Chunking
│   ├── embeddings.py      # HuggingFace embeddings
│   ├── vectorstore.py     # FAISS build / save / search
│   └── rag_chain.py       # Gemini RAG answers
└── faiss_index/           # Created after first upload (gitignored)
```

## Setup

### 1. Clone or open the project

```bash
cd college-rag-assistant
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The first run downloads the HuggingFace embedding model (~90 MB).

### 4. Configure Gemini API key

```bash
copy .env.example .env
```

Edit `.env` and set your key from [Google AI Studio](https://aistudio.google.com/apikey):

```
GOOGLE_API_KEY=your_actual_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## How to use

1. In the **sidebar**, upload one or more PDF files.
2. Click **Process PDFs** to build the vector database.
3. Type questions in the chat box at the bottom.
4. Expand **Source chunks used** under each answer to see which PDF passages were retrieved.

## Requirements

- Python 3.10+
- Internet for Gemini API (embeddings run locally after first download)

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `GOOGLE_API_KEY is missing` | Create `.env` from `.env.example` and add your key |
| No text extracted | PDF may be scanned images—use OCR PDFs or text-based PDFs |
| Slow first question | Embedding model loads on first use; later queries are faster |

## License

MIT — use freely for learning and college projects.
