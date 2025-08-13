
import os
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

from rag.indexing import build_or_load_index, persist_index, load_index_only
from rag.retriever import get_retriever_nodes
from llm.generator import answer_with_context
from utils.pubmed import fetch_pubmed_abstracts

load_dotenv()

st.set_page_config(
    page_title="Medical QA — BioMistral + RAG",
    page_icon="🩺",
    layout="wide"
)

# ---------- Minimal styling (cards, badges) ----------
st.markdown(
    """
    <style>
    .source-card {
        padding: 0.75rem 1rem;
        border: 1px solid #e6e6e6;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,.04);
        margin-bottom: 0.75rem;
        background: #fff;
    }
    .badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        font-size: 0.75rem;
        border: 1px solid #ddd;
        margin-right: 0.25rem;
        background: #f7f7f8;
    }
    .answer-card {
        padding: 1rem 1.25rem;
        border-radius: 14px;
        background: #f9fafb;
        border: 1px solid #eef0f3;
    }
    .small-muted { color: #6b7280; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩺 Medical QA Assistant — BioMistral + RAG")
st.caption("**Educational purposes only. Not a substitute for professional medical advice.**")

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("📥 Build Knowledge Base")
    storage_dir = st.text_input("Index storage directory", value=".storage")
    chunk_size = st.slider("Chunk size", 256, 2048, 1024, step=64)
    chunk_overlap = st.slider("Chunk overlap", 0, 512, 100, step=10)
    top_k = st.slider("Top-k passages", 1, 10, 4, step=1)
    temperature = st.slider("LLM temperature", 0.0, 1.5, 0.2, step=0.05)
    
    uploaded_pdfs = st.file_uploader(
        "Upload clinical PDFs", type=["pdf"], accept_multiple_files=True
    )
    url_text = st.text_area("Add web URLs (one per line)", height=90, placeholder="https://www.who.int/...\nhttps://www.cdc.gov/...")
    
    with st.expander("➕ Import PubMed abstracts"):
        pm_query = st.text_input("PubMed search term", placeholder="e.g., type 2 diabetes metformin RCT")
        pm_count = st.slider("Max articles", 0, 50, 10, step=5)
        do_pubmed = st.checkbox("Fetch on build", value=False)
    
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        rebuild = st.button("🔨 Rebuild / Extend Index")
    with col_sb2:
        reset = st.button("🗑️ Reset Storage (fresh)")

# Session state for index
if "index" not in st.session_state:
    st.session_state.index = None

if reset:
    import shutil
    if os.path.isdir(storage_dir):
        shutil.rmtree(storage_dir)
    st.session_state.index = None
    st.success("Storage cleared. Rebuild the index.")

if rebuild:
    with st.spinner("Building index..."):
        urls = [u.strip() for u in (url_text or "").splitlines() if u.strip()]
        pubmed_docs = []
        if do_pubmed and pm_query and pm_count > 0:
            st.info("Fetching PubMed abstracts...")
            pubmed_docs = fetch_pubmed_abstracts(pm_query, pm_count)
            st.write(f"Fetched {len(pubmed_docs)} PubMed abstracts.")
        
        st.session_state.index = build_or_load_index(
            storage_dir=storage_dir,
            pdf_files=uploaded_pdfs or [],
            urls=urls,
            extra_docs=pubmed_docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        persist_index(st.session_state.index, storage_dir)
    st.success("Index ready!")

# Load existing index if available
if st.session_state.index is None and os.path.isdir(storage_dir):
    try:
        st.session_state.index = load_index_only(storage_dir)
    except Exception:
        pass

# ---------- Query UI ----------
st.subheader("Ask a medical question")
query = st.text_input("Your question", placeholder="e.g., What are first-line treatments for hypertension in adults?")
col_actions = st.columns([1,1,4])
with col_actions[0]:
    run = st.button("💡 Generate Answer")
with col_actions[1]:
    show_sources_only = st.button("🔎 Retrieve Sources")

# ---------- Run retrieval / generation ----------
if run or show_sources_only:
    if st.session_state.index is None:
        st.warning("Please build the index first (upload files and/or add URLs)")
    elif not query.strip():
        st.warning("Enter a question to proceed.")
    else:
        with st.spinner("Retrieving passages..."):
            nodes = get_retriever_nodes(st.session_state.index, query, top_k=top_k)
        
        if show_sources_only:
            st.info(f"Retrieved {len(nodes)} passages.")
        else:
            with st.spinner("Thinking with BioMistral..."):
                answer, used_sources = answer_with_context(
                    query=query,
                    nodes=nodes,
                    temperature=temperature
                )
            st.markdown('<div class="answer-card">', unsafe_allow_html=True)
            st.markdown(answer)
            st.markdown('</div>', unsafe_allow_html=True)

        # ---------- Sources panel ----------
        st.subheader("Sources")
        for i, n in enumerate(nodes, 1):
            meta = n.metadata or {}
            title = meta.get("title") or meta.get("file_name") or meta.get("source") or "Document"
            score = getattr(n, "score", None)
            score_txt = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
            st.markdown(f"**{i}. {title}**  \n<span class='small-muted'>Similarity: {score_txt}</span>", unsafe_allow_html=True)
            with st.expander("View passage"):
                st.markdown(f"<div class='source-card'>{n.text}</div>", unsafe_allow_html=True)
            # Show metadata badges
            md_badges = []
            if isinstance(meta, dict):
                for k, v in meta.items():
                    if v is None:
                        continue
                    sv = str(v)
                    if len(sv) > 80:
                        sv = sv[:77] + "..."
                    md_badges.append(f"<span class='badge'><b>{k}:</b> {sv}</span>")
            if md_badges:
                st.markdown(" ".join(md_badges), unsafe_allow_html=True)

st.divider()
with st.expander("⚙️ Environment & Model"):
    st.write("Using Hugging Face Inference API with model specified by `HF_MODEL_ID` (default: BioMistral/BioMistral-7B).")
    st.code(f"HF_INFERENCE_URL={os.getenv('HF_INFERENCE_URL','')}
HF_MODEL_ID={os.getenv('HF_MODEL_ID','BioMistral/BioMistral-7B')}", language="bash")
    st.caption("Set these in your `.env`.")
