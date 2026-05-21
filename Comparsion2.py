import time
import pandas as pd
import nltk
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from bert_score import score
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

# --- Initialization ---
nltk.download('punkt')
df = pd.read_csv('cleaned_university_faqs.csv')
test_set = df.sample(50) # 30-50 questions as per methodology [cite: 57]
knowledge_base = df['target'].tolist()

# 1. Setup BM25
tokenized_kb = [word_tokenize(doc.lower()) for doc in knowledge_base]
bm25 = BM25Okapi(tokenized_kb)

# 2. Setup RAG
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local("university_faiss_index_folder", embeddings, allow_dangerous_deserialization=True)
llm = Ollama(model="qwen:4b")
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

def run_comparison():
    results = []
    
    for _, row in test_set.iterrows():
        query = row['input']
        reference = row['target']
        
        # BM25 Evaluation
        start = time.time()
        tokenized_query = word_tokenize(query.lower())
        bm25_res = bm25.get_top_n(tokenized_query, knowledge_base, n=1)[0]
        bm25_time = time.time() - start
        
        # RAG Evaluation [cite: 36, 55]
        start = time.time()
        rag_res = qa_chain.invoke(query)["result"]
        rag_time = time.time() - start
        
        results.append({
            "query": query,
            "reference": reference,
            "bm25_output": bm25_res,
            "rag_output": rag_res,
            "bm25_latency": bm25_time,
            "rag_latency": rag_time
        })

    eval_df = pd.DataFrame(results)

    # --- Calculate BERTScore ---
    # P = Precision, R = Recall, F1 = F1-Score (Semantic Quality)
    P_b, R_b, F1_bm25 = score(eval_df['bm25_output'].tolist(), eval_df['reference'].tolist(), lang="en")
    P_r, R_r, F1_rag = score(eval_df['rag_output'].tolist(), eval_df['reference'].tolist(), lang="en")

    # --- Final Report ---
    print("\n--- Final Comparative Analysis ---")
    summary = {
        "Metric": ["Avg BERTScore (F1)", "Avg Latency (s)"],
        "BM25 (Legacy)": [f"{F1_bm25.mean():.4f}", f"{eval_df['bm25_latency'].mean():.4f}"],
        "RAG (Advanced)": [f"{F1_rag.mean():.4f}", f"{eval_df['rag_latency'].mean():.4f}"]
    }
    print(pd.DataFrame(summary))

if __name__ == "__main__":
    run_comparison()