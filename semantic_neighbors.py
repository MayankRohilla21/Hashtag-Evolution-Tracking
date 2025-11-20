# semantic_neighbors_keywords.py
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

df = pd.read_csv("reddit_politics_cleaned.csv")

# Convert clean_tokens column from string → list
df['clean_tokens'] = df['clean_tokens'].apply(eval)

# Use clean tokens (important!)
texts = [" ".join(tokens) for tokens in df["clean_tokens"]]

print("Extracting keywords from dataset...")

vectorizer = TfidfVectorizer(
    max_features=200,
    stop_words="english",  # safe
    ngram_range=(1,1)
)

X = vectorizer.fit_transform(texts)
keywords = vectorizer.get_feature_names_out()

print("Total extracted keywords:", len(keywords))

print("Loading embedding model...")
model = SentenceTransformer("all-mpnet-base-v2")

print("Embedding all keywords...")
keyword_embeddings = {
    kw: model.encode(kw) for kw in keywords
}

def get_similar_keywords(target, num=5):
    if target not in keyword_embeddings:
        print(f"⚠ Keyword '{target}' not found in dataset.")
        return []

    target_emb = keyword_embeddings[target].reshape(1, -1)
    similarities = []

    for kw, emb in keyword_embeddings.items():
        if kw == target:
            continue
        score = cosine_similarity(target_emb, emb.reshape(1, -1))[0][0]
        similarities.append((kw, score))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:num]

# Example usage
if __name__ == "__main__":
    query = "gaslighting"
    print("\nTop similar keywords to:", query)
    results = get_similar_keywords(query, num=10)

    for word, sim in results:
        print(f"{word} — similarity={sim:.4f}")
