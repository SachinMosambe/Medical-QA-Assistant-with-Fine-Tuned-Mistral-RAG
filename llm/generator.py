
from typing import List, Tuple
from llama_index.core.schema import NodeWithScore
from llm.biomistral_client import BioMistralClient

SYS_PROMPT = (
    "You are a careful medical assistant. Use only the provided context to answer.\n"
    "Cite sources inline as [S1], [S2], ... based on the order of passages.\n"
    "If the answer is not in the context, say you don't know and suggest what to search.\n"
    "Do not provide medical advice; this is for educational purposes only."
)

def _build_context_block(nodes: List[NodeWithScore]) -> str:
    parts = []
    for i, n in enumerate(nodes, 1):
        meta = n.metadata or {}
        title = meta.get("title") or meta.get("file_name") or meta.get("source") or "Document"
        parts.append(f"[S{i}] {title}\n{n.text}")
    return "\n\n".join(parts)

def _build_prompt(query: str, nodes: List[NodeWithScore]) -> str:
    ctx = _build_context_block(nodes)
    prompt = (
        f"{SYS_PROMPT}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    return prompt

def answer_with_context(query: str, nodes: List[NodeWithScore], temperature: float = 0.2) -> Tuple[str, List[int]]:
    client = BioMistralClient()
    prompt = _build_prompt(query, nodes)
    output = client.generate(prompt=prompt, temperature=temperature, max_new_tokens=600)
    return output, list(range(1, len(nodes)+1))
