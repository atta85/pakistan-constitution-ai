# 🇵🇰 Constitution of Pakistan AI Assistant

A Retrieval-Augmented Generation (RAG) application for question answering over the Constitution of the Islamic Republic of Pakistan.

## Components

- Streamlit — web application
- FAISS — vector similarity search
- BAAI/bge-small-en-v1.5 — embeddings
- Groq API — LLM inference
- Python — application backend

## Repository structure

```text
pakistan-constitution-ai/
├── app.py
├── constitution.faiss
├── metadata.json
├── config.json
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

## Architecture

```text
Constitution PDF
      ↓
Text extraction + chunking
      ↓
BGE embeddings
      ↓
FAISS vector index
      ↓
Streamlit
      ↓
Question embedding
      ↓
Semantic retrieval
      ↓
Retrieved constitutional passages
      ↓
Groq LLM
      ↓
Grounded answer + source passages
```

## Groq API key

Never commit the Groq API key to GitHub.

For Streamlit Community Cloud, add this to the app's Secrets:

```toml
GROQ_API_KEY = "your_api_key_here"
```

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

The FAISS index and metadata must remain synchronized. Do not reorder the records in `metadata.json` without rebuilding the FAISS index.

The application is an AI research/information assistant and is not a substitute for professional legal advice or authoritative legal interpretation.
