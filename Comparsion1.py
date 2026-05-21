import time
import pandas as pd
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- Setup Evaluation Data ---
df = pd.read_csv('cleaned_university_faqs.csv')
# Use the 'input' column as our test queries and 'target' as ground truth
test_set = df.sample(30)  # Selecting 30 questions as per methodology 
knowledge_base = df['target'].tolist()

# --- Initialize Systems ---
# 1. BM25 Baseline
tokenized_kb = [word_tokenize(doc.lower()) for doc in knowledge_base]
bm25 = BM25Okapi(tokenized_kb)

# 2. RAG Retriever (FAISS) [cite: 53]
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local("university_faiss_index_folder", embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

def evaluate_systems():
    results = []

    for _, row in test_set.iterrows():
        query = row['input']
        ground_truth = row['target']
        
        # --- Evaluate BM25 ---
        start_time = time.time()
        tokenized_query = word_tokenize(query.lower())
        bm25_hits = bm25.get_top_n(tokenized_query, knowledge_base, n=3)
        bm25_latency = time.time() - start_time
        
        bm25_hit1 = 1 if bm25_hits[0] == ground_truth else 0
        bm25_hit3 = 1 if ground_truth in bm25_hits else 0

        # --- Evaluate RAG Retrieval ---
        start_time = time.time()
        rag_docs = retriever.invoke(query)
        rag_hits = [doc.page_content for doc in rag_docs]
        rag_latency = time.time() - start_time
        
        rag_hit1 = 1 if rag_hits[0] == ground_truth else 0
        rag_hit3 = 1 if ground_truth in rag_hits else 0

        results.append({
            "query": query,
            "bm25_hit@1": bm25_hit1, "bm25_hit@3": bm25_hit3, "bm25_latency": bm25_latency,
            "rag_hit@1": rag_hit1, "rag_hit@3": rag_hit3, "rag_latency": rag_latency
        })

    # --- Generate Final Metrics Report ---
    report_df = pd.DataFrame(results)
    
    # Calculate Hit@K Percentages [cite: 58]
    summary = {
        "Metric": ["Hit@1 Accuracy", "Hit@3 Accuracy", "Avg Latency (s)"],
        "BM25 (Legacy)": [
            f"{(report_df['bm25_hit@1'].mean()*100):.2f}%",
            f"{(report_df['bm25_hit@3'].mean()*100):.2f}%",
            f"{report_df['bm25_latency'].mean():.4f}"
        ],
        "RAG (Advanced)": [
            f"{(report_df['rag_hit@1'].mean()*100):.2f}%",
            f"{(report_df['rag_hit@3'].mean()*100):.2f}%",
            f"{report_df['rag_latency'].mean():.4f}"
        ]
    }
    
    return pd.DataFrame(summary)

# Execute Comparison
comparison_report = evaluate_systems()
print(comparison_report)