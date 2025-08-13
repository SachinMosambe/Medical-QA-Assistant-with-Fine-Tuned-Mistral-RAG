
from typing import List
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

def get_retriever_nodes(index: VectorStoreIndex, query: str, top_k: int = 4) -> List[NodeWithScore]:
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return nodes
