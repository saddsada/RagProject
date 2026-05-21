import pandas as pd
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Ensure tokenization resources are available for the BM25 algorithm [cite: 307]
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def run_baseline_system():
    # 1. Load the cleaned knowledge base [cite: 307]
    # Based on the uploaded dataset, the columns are 'input' (questions) and 'target' (answers)
    try:
        df = pd.read_csv('cleaned_university_faqs.csv')
        # We use the 'target' column as the knowledge corpus to be retrieved [cite: 279, 311]
        corpus = df['target'].tolist()
    except FileNotFoundError:
        print("Error: 'cleaned_university_faqs.csv' not found.")
        return
    except KeyError:
        print("Error: The CSV must contain 'input' and 'target' columns.")
        return

    # 2. Tokenize the corpus for BM25 [cite: 308, 310]
    # This matches the probabilistic retrieval paradigm modeled by BM25 [cite: 267]
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]

    # 3. Initialize the BM25 Baseline System [cite: 310]
    bm25 = BM25Okapi(tokenized_corpus)

    print("--- Baseline System (BM25 Keyword Search) Loaded ---")
    
    # 4. Define the retrieval function
    def get_baseline_response(query, n=1):
        """
        Performs keyword matching to find the most relevant document snippet[cite: 311].
        """
        tokenized_query = word_tokenize(query.lower())
        
        # Retrieve the top-n most relevant snippets [cite: 267]
        results = bm25.get_top_n(tokenized_query, corpus, n=n)
        
        return results[0] if results else "No relevant information found."

    # 5. Test the system
    # Example query based on the AI history content in your CSV
    test_query = "Who built the first neural network?"
    print(f"\nUser Query: {test_query}")
    print(f"BM25 Response: {get_baseline_response(test_query)}")

if __name__ == "__main__":
    run_baseline_system()