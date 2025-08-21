from langchain.utilities import SerpAPIWrapper
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

class WebRetriever:
    def __init__(self, serpapi_key):
        self.search = SerpAPIWrapper(api_key=serpapi_key)
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def retrieve_and_index(self, query):
        search_results = self.search.run(query)
        urls = self._extract_urls(search_results)
        docs = []
        for url in urls:
            loader = WebBaseLoader(url)
            try:
                docs.extend(loader.load())
            except Exception as e:
                print(f"Failed to load {url}: {e}")

        # Create vectorstore index
        vectorstore = FAISS.from_documents(docs, self.embedding_model)
        return vectorstore

    def _extract_urls(self, text):
        import re
        urls = re.findall(r'https?://\S+', text)
        return list(set(urls))
