
import html
import requests
from typing import List
from llama_index.core import Document

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def _esearch(query: str, retmax: int = 10) -> List[str]:
    params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": retmax, "sort": "relevance"}
    r = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    return js.get("esearchresult", {}).get("idlist", [])

def _efetch(pmids: List[str]) -> List[dict]:
    if not pmids:
        return []
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    r = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    from xml.etree import ElementTree as ET
    root = ET.fromstring(r.text)
    out = []
    for art in root.findall(".//PubmedArticle"):
        title_el = art.find(".//ArticleTitle")
        abst_el = art.find(".//Abstract/AbstractText")
        title = "".join(title_el.itertext()).strip() if title_el is not None else "PubMed Article"
        abstract = "".join(abst_el.itertext()).strip() if abst_el is not None else ""
        out.append({"title": html.unescape(title), "abstract": html.unescape(abstract)})
    return out

def fetch_pubmed_abstracts(query: str, max_n: int = 10) -> List[Document]:
    """Return LlamaIndex Documents built from PubMed titles+abstracts."""
    pmids = _esearch(query, retmax=max_n)
    arts = _efetch(pmids)
    docs = []
    for a in arts:
        if not a.get("abstract"):
            continue
        docs.append(Document(text=f"{a['title']}\n\n{a['abstract']}", metadata={"title": a["title"], "source": "PubMed"}))
    return docs
