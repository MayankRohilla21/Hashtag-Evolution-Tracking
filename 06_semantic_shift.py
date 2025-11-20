import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load cleaned data
df = pd.read_csv("reddit_politics_cleaned.csv", parse_dates=['date']).dropna(subset=['date'])
print("Available columns:", df.columns.tolist())

# Convert clean_tokens column from string -> list
df['clean_tokens'] = df['clean_tokens'].apply(eval)

# Choose your keyword
KEYWORD = "gaslighting"

# Define time windows
df['year'] = df['date'].dt.year

# Filter posts containing keyword (in cleaned tokens)
df = df[df['clean_tokens'].apply(lambda toks: KEYWORD.lower() in toks)]

print("Posts mentioning", KEYWORD, ":", len(df))

# Load Sentence-BERT
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert clean tokens back into sentence-like strings for embedding
texts = [" ".join(tokens) for tokens in df['clean_tokens']]

embeddings = model.encode(texts, show_progress_bar=True)

# Add embeddings back
df['embedding'] = list(embeddings)

# Average embedding per year
yearly_vectors = {}
for year, group in df.groupby('year'):
    yearly_vectors[year] = np.mean(list(group['embedding']), axis=0)

# Compare years pairwise
years = sorted(yearly_vectors.keys())
print("\nCosine similarity between years:")
for i in range(len(years) - 1):
    sim = cosine_similarity(
        [yearly_vectors[years[i]]],
        [yearly_vectors[years[i+1]]]
    )[0][0]
    print(f"{years[i]} -> {years[i+1]}: {sim:.4f}")

# Plot similarity over time
if len(years) > 1:
    sims = [
        cosine_similarity([yearly_vectors[y1]], [yearly_vectors[y2]])[0][0]
        for y1, y2 in zip(years[:-1], years[1:])
    ]

    plt.figure(figsize=(8, 4))
    plt.plot(years[1:], sims, marker='o')
    plt.title(f"Semantic similarity of '{KEYWORD}' over time")
    plt.xlabel("Year")
    plt.ylabel("Cosine similarity to previous year")
    plt.tight_layout()
    plt.savefig("semantic_shift_politics.png")
    plt.show()

    print("Semantic shift plot saved as semantic_shift_politics.png")
else:
    print("Not enough years of data to plot semantic shift")
