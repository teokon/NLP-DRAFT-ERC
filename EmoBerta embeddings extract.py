# -*- coding: utf-8 -*-
"""Untitled12.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BVv4mQ-5B1KZdYArrQn0S3zhLraR1vtE
"""

!pip install transformers datasets
!pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
!pip install torch-geometric
from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd

# List the files you’ve just uploaded to verify their names
print(os.listdir('/content'))

# Adjust these filenames if needed to match exactly what you see
train_path = '/content/train_sent_emo_cleaned_processed.csv'
dev_path   = '/content/dev_sent_emo_cleaned_processed.csv'
test_path  = '/content/test_sent_emo_cleaned_processed.csv'

# Load into pandas
train_df = pd.read_csv(train_path)
dev_df   = pd.read_csv(dev_path)
test_df  = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Dev   shape:", dev_df.shape)
print("Test  shape:", test_df.shape)
train_df.head()

# Load model directly
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("tae898/emoberta-base")
emoberta = AutoModel.from_pretrained("tae898/emoberta-base")

train_df = train_df.rename(columns={
    'Utterance': 'text',
    'Emotion': 'label',
    'Sentiment': 'sentiment_label',
    'Dialogue_ID': 'dialogue_id',
    'Utterance_ID': 'utterance_id'
})
# Same for dev and test
for df in [dev_df, test_df]:
    df.rename(columns={
        'Utterance': 'text',
        'Emotion': 'label',
        'Sentiment': 'sentiment_label',
        'Dialogue_ID': 'dialogue_id',
        'Utterance_ID': 'utterance_id'
    }, inplace=True)

# Sort each set by dialogue and turn order
train_df.sort_values(['dialogue_id', 'utterance_id'], inplace=True)
dev_df.sort_values(['dialogue_id', 'utterance_id'], inplace=True)
test_df.sort_values(['dialogue_id', 'utterance_id'], inplace=True)

# Verify
display(train_df[['dialogue_id','utterance_id','text','label','sentiment_label']].head(10))

import torch

train_texts = train_df['text'].tolist()     # e.g. ["also I was the point person...", "You must’ve had...", ...]
dev_texts = dev_df['text'].tolist()
test_texts = test_df['text'].tolist()
# 3. Define a batching helper
def get_utterance_embeddings(texts, batch_size=32):
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            outputs = emoberta(**enc)
            # CLS token is at position 0
            cls_emb = outputs.last_hidden_state[:, 0, :]    # shape [batch, hidden_dim]
            all_embs.append(cls_emb.cpu())
    return torch.cat(all_embs, dim=0)  # [total_utterances, hidden_dim]

# 4. Run it
train_embeddings = get_utterance_embeddings(train_texts)
dev_embeddings = get_utterance_embeddings(dev_texts)
test_embeddings = get_utterance_embeddings(test_texts)
print("Train embeddings shape:", train_embeddings.shape)

import os
import numpy as np
import pandas as pd

# Directory where you want to save your pickles
output_dir = '/content/drive/MyDrive/MELD'
os.makedirs(output_dir, exist_ok=True)

# List of (split name, DataFrame, embeddings tensor)
splits = [
    ('train', train_df, train_embeddings),
    ('dev',   dev_df,   dev_embeddings),
    ('test',  test_df,  test_embeddings),
]

for name, df, emb_tensor in splits:
    # 1) Convert to NumPy
    emb_np = emb_tensor.cpu().numpy()   # shape (Ni, 768)

    # 2) Break into list of 1D arrays
    emb_list = [emb_np[i] for i in range(emb_np.shape[0])]

    # 3) Attach as a new column
    df['emb_array'] = emb_list

    # 4) Pickle the DataFrame
    path = os.path.join(output_dir, f'{name}_with_emb.pkl')
    df.to_pickle(path)
    print(f"Saved {name} split ({len(df)} rows) → {path}")

import pandas as pd

# 1. Load the pickled DataFrame
df = pd.read_pickle('/content/drive/MyDrive/MELD/train_with_emb.pkl')

# 2. Peek at the first few rows (drops the array column for readability)
print(df.drop(columns=['emb_array']).head())

# 3. Check dtypes to confirm emb_array is object (NumPy arrays)
print(df.dtypes)

# 4. Inspect one embedding
arr0 = df.loc[0, 'emb_array']
print(type(arr0), arr0.shape)   # should be <class 'numpy.ndarray'>, e.g. (768,)

# 5. View a slice of that array
print(arr0[:10])                # first 10 dimensions

for idx, row in df.head(3).iterrows():
    print(f"Row {idx}: label={row['sentiment_label']}, emb_array[:5]={row['emb_array'][:5]}")
