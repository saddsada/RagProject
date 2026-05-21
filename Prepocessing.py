import json
import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str): return ""
    # 1. Strip HTML tags
    text = re.sub(r'<.*?>', '', text)
    # 2. Standardize whitespace and remove newlines
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Step 1: Load and Process JSON (Intents) ---
with open('RawFaqDataset.json', 'r' ,encoding='utf-8') as f:
    data_json = json.load(f)

json_list = []
for intent in data_json.get('intents', []):
    # Combine all possible responses into one 'Source of Truth'
    full_response = " ".join(intent.get('responses', []))
    for pattern in intent.get('patterns', []):
        json_list.append({
            'input': clean_text(pattern),
            'target': clean_text(full_response)
        })
df_json = pd.DataFrame(json_list)

# --- Step 2: Load and Process CSV ---
df_csv = pd.read_csv('combined_university_faq.csv', encoding='utf-8')
df_csv['input'] = df_csv['input'].apply(clean_text)
df_csv['target'] = df_csv['target'].apply(clean_text)

# --- Step 3: Combine and Deduplicate ---
combined_df = pd.concat([df_csv, df_json], ignore_index=True)
# Only keep the first version of any specific question
combined_df = combined_df.drop_duplicates(subset=['input'], keep='first')

# --- Step 4: Export ---
combined_df.to_csv('cleaned_university_faq.csv', index=False)

print(f"Final Dataset Size: {len(combined_df)} unique rows.")