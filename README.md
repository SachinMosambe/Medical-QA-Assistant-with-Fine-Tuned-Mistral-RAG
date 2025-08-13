# Medical QA Assistant — BioMistral + LlamaIndex RAG + Streamlit

A complete, local-first RAG app for medical Q&A built with:
- **BioMistral-7B** via **Hugging Face Inference API**
- **LlamaIndex** for chunking, embedding (MiniLM), vector indexing, and retrieval
- **Streamlit** frontend with side-by-side sources
- PDF uploads, PubMed abstracts fetch, and general web URLs ingestion

> ⚠️ This app is for **educational research** only. It does **not** provide medical advice.

## Quickstart

1. **Clone / unzip** this project.
2. Create a virtualenv and install deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate  # (Windows) .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and set `HF_API_TOKEN`. Optionally set `HF_INFERENCE_URL` if you have a private endpoint.
4. Run:
   ```bash
   streamlit run app.py
   ```

## Project Layout

```text
medical_qa_mistral_rag/
├─ app.py                      # Streamlit UI
├─ requirements.txt
├─ .env.example
├─ README.md
├─ rag/
│  ├─ __init__.py
│  ├─ indexing.py              # loaders, chunking, index build/load
│  └─ retriever.py             # query-time retrieval
├─ llm/
│  ├─ __init__.py
│  ├─ biomistral_client.py     # HF Inference client wrapper
│  └─ generator.py             # prompt + generation
└─ utils/
   ├─ __init__.py
   └─ pubmed.py                # fetch PubMed abstracts (E-utilities)
```

## Notes

- Embeddings use `sentence-transformers/all-MiniLM-L6-v2` via LlamaIndex HuggingFaceEmbedding.
- Vector index persisted to `.storage` so you don’t re-embed every run.
- Set the number of top-k nodes, chunk size/overlap, and temperature in the UI.