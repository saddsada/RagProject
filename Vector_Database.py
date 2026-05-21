import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def create_vector_database():
    # 1. Load the cleaned knowledge base [cite: 53]
    try:
        df = pd.read_csv('cleaned_university_faqs.csv')
        # We use the 'target' column as the knowledge source [cite: 12]
        texts = df['target'].tolist()
    except FileNotFoundError:
        print("Error: 'cleaned_university_faqs.csv' not found. Please ensure the file is in your directory.")
        return

    # 2. Initialize the Embedding Model [cite: 32, 53]
    # We use 'all-MiniLM-L6-v2' for efficient, high-dimensional text representations.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Generating embeddings and building the FAISS index...")

    # 3. Create the Vector Store from the 'target' text [cite: 33, 53]
    # This process converts each text fragment into a sequence of vectors.
    vector_store = FAISS.from_texts(texts, embeddings)

    # 4. Save correctly for the LangChain Retrieval Chain [cite: 55]
    # This creates a folder containing 'index.faiss' and 'index.pkl'
    vector_store.save_local("university_faiss_index_folder")

    print("Step 3 Complete!")
    print(f"Vector Database created with {len(texts)} entries in 'university_faiss_index_folder'.")

if __name__ == "__main__":
    create_vector_database()