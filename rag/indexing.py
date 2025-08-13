
import os
from typing import List, Optional
from io import BytesIO

from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings, load_index_from_storage
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.file import PyMuPDFReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set global embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def _pdf_docs_from_uploads(files) -> List[Document]:
    """Convert Streamlit UploadedFile list to LlamaIndex Documents."""
    docs = []
    if not files:
        return docs
    reader = PyMuPDFReader()
    for f in files:
        b = f.read()
        f.seek(0)
        tmp = BytesIO(b)
        file_docs = reader.load_data(file=tmp, extra_info={"file_name": f.name})
        docs.extend(file_docs)
    return docs

def _docs_from_urls(urls: List[str]) -> List[Document]:
    if not urls:
        return []
    web_docs = SimpleWebPageReader(html_to_text=True).load_data(urls)
    return web_docs

def build_or_load_index(
    storage_dir: str = ".storage",
    pdf_files=None,
    urls: Optional[List[str]] = None,
    extra_docs: Optional[List[Document]] = None,
    chunk_size: int = 1024,
    chunk_overlap: int = 100
) -> VectorStoreIndex:
    os.makedirs(storage_dir, exist_ok=True)
    
    pdf_docs = _pdf_docs_from_uploads(pdf_files)
    url_docs = _docs_from_urls(urls or [])
    docs = []
    docs.extend(pdf_docs)
    docs.extend(url_docs)
    if extra_docs:
        docs.extend(extra_docs)
    
    from llama_index.core.node_parser import SentenceSplitter
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(docs)

    if os.listdir(storage_dir):
        # Extend existing index
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
        index.insert_nodes(nodes)
    else:
        index = VectorStoreIndex(nodes, show_progress=True)
    return index

def persist_index(index: VectorStoreIndex, storage_dir: str = ".storage"):
    index.storage_context.persist(persist_dir=storage_dir)

def load_index_only(storage_dir: str = ".storage") -> VectorStoreIndex:
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    return load_index_from_storage(storage_context)
