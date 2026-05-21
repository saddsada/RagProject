import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

def run_rag_system():
    # 1. Setup Embeddings [cite: 12, 53]
    # Must match the model used to create the database in Step 3.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Load the Vector Database [cite: 12, 53]
    # This loads 'index.faiss' and 'index.pkl' from your local folder.
    index_path = "university_faiss_index_folder"
    if not os.path.exists(index_path):
        print(f"Error: Folder '{index_path}' not found. Please run Step 3 first.")
        return

    vector_store = FAISS.load_local(
        index_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    # 3. Initialize the LLM (Ollama Qwen:4b) [cite: 55]
    # Ensure Ollama is running and you have run 'ollama pull qwen:4b'.
    llm = Ollama(model="qwen:4b")

    # 4. Define the Prompt Template [cite: 7, 35]
    # Shifts from passive document matching to active, context-aware synthesis.
    template = """
    You are an intelligent university educational assistant. 
    Use the following pieces of retrieved context to answer the student's question.
    If the context does not contain the answer, politely state that you do not have that information.
    Keep the answer factual and concise based on the provided documents.

    Context: {context}
    Question: {question}

    Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # 5. Create the RetrievalQA Chain [cite: 35, 54]
    # Orchestrates the dense vector retrieval and the generative model.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    print("--- University RAG System (Qwen:4b) Ready ---")
    
    # 6. Test the System [cite: 40]
    query = "What is the first neural network called?"
    print(f"\nStudent Query: {query}")
    
    print("Generating response...")
    response = qa_chain.invoke(query)
    
    print(f"RAG Answer: {response['result']}")

if __name__ == "__main__":
    run_rag_system()