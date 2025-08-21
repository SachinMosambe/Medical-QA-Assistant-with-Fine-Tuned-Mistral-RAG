import streamlit as st
from src.retriever import WebRetriever
from src.generator import MedicalLLM
from src.rag_chain import build_rag_chain

# Replace with your actual SerpAPI key & model paths
SERPAPI_API_KEY = "your_serpapi_key"
BASE_MODEL_NAME = "unsloth/mistral-7b-v0.3-4bit"
ADAPTER_PATH = "path/to/your/lora_adapter"

@st.cache_resource(show_spinner=False)
def load_components():
    retriever = WebRetriever(SERPAPI_API_KEY)
    generator = MedicalLLM(BASE_MODEL_NAME, ADAPTER_PATH)
    return retriever, generator

def main():
    st.title("Medical Chatbot with RAG & Mistral 7B")
    retriever, generator = load_components()

    if "history" not in st.session_state:
        st.session_state.history = []

    query = st.text_input("Ask a medical question:")

    if query:
        with st.spinner("Searching and generating answer..."):
            vectorstore = retriever.retrieve_and_index(query)
            rag_chain = build_rag_chain(generator.get_llm(), vectorstore)
            answer = rag_chain.run(query)

        st.session_state.history.append(("User", query))
        st.session_state.history.append(("Bot", answer))

    for sender, message in st.session_state.history:
        if sender == "User":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Bot:** {message}")

if __name__ == "__main__":
    main()
