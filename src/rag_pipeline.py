from langchain.chains import RetrievalQA

def build_rag_chain(llm, retriever):
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever.as_retriever(), chain_type="stuff")
