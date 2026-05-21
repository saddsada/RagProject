import streamlit as st
import pandas as pd
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import nltk

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Page Configuration ---
st.set_page_config(page_title="EduBot: BM25 vs RAG", layout="wide")
st.title("🎓 Educational Chatbot Evaluation")
st.markdown("Compare **Legacy BM25** keyword search against **Advanced RAG** semantic generation.")

# --- System Initialization ---
@st.cache_resource
def initialize_systems():
    # Load the new dataset structure
    df = pd.read_csv('cleaned_university_faqs.csv')
    # Use 'target' as the authoritative knowledge base
    knowledge_base = df['target'].tolist()
    
    # 1. Setup BM25 (Baseline)
    tokenized_kb = [word_tokenize(doc.lower()) for doc in knowledge_base]
    bm25_system = BM25Okapi(tokenized_kb)
    
    # 2. Setup RAG (Advanced)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        "university_faiss_index_folder", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    llm = Ollama(model="qwen:4b")
    
    # Define custom prompt to shift to context-aware synthesis
    template = """Use the following context to answer the question. 
    If you don't know, say you don't know. 
    Context: {context}
    Question: {question}
    Answer:"""
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PromptTemplate.from_template(template)}
    )
    
    return bm25_system, knowledge_base, qa_chain

# Load systems once
bm25, corpus, rag_pipeline = initialize_systems()

# --- User Interface ---
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Retrieval Mode:", ("BM25 (Baseline)", "RAG (Advanced)"))
    st.info("BM25 matches keywords. RAG synthesizes answers.")

query = st.text_input("Enter your question:", placeholder="e.g., Who created the Hebbian learning rule?")

if query:
    if mode == "BM25 (Baseline)":
        st.subheader("Results from BM25 (Legacy System)")
        tokenized_query = word_tokenize(query.lower())
        results = bm25.get_top_n(tokenized_query, corpus, n=1)
        
        if results:
            st.info(results[0])
            st.caption("Output: Exact snippet match from the document.")
        else:
            st.warning("No keyword matches found.")

    else:
        st.subheader("Results from RAG (LLM Generation)")
        with st.spinner("Qwen:4b is generating an answer..."):
            response = rag_pipeline.invoke(query)
            st.success(response["result"])
            st.caption("Output: Context-aware synthesis using Qwen:4b.")

# --- Footer for Thesis Progress ---
st.divider()
st.write(f"Current Phase: **Step 5: Streamlit Interface**")