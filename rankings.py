# rankings.py

import pandas as pd
import numpy as np

# Import from your FIXED semantic_neighbors file
from semantic_neighbors import get_similar_keywords, keyword_embeddings

try:
    stability_df = pd.read_csv("keyword_stability.csv")
    print("Loaded keyword_stability.csv")
except:
    print("WARNING: keyword_stability.csv not found — stability set to 0 for all keywords.")
    stability_df = pd.DataFrame(columns=["keyword", "year", "stability_score"])

def get_stability(keyword):
    rows = stability_df[stability_df["keyword"] == keyword]

    if len(rows) == 0:
        return 0.0

    return rows["stability_score"].mean()

def compute_score(similarity, stability):
    return 0.7 * similarity + 0.3 * stability

def recommend_keywords(target_keyword, top_k=5):

    candidates = get_similar_keywords(target_keyword, num=20)

    ranked = []
    for kw, sim in candidates:
        stab = get_stability(kw)
        final_score = compute_score(sim, stab)
        ranked.append((kw, sim, stab, final_score))

    ranked.sort(key=lambda x: x[3], reverse=True)

    return ranked[:top_k]

# Example usage
if __name__ == "__main__":
    target = "gaslighting"

    print(f"\nTop recommended replacements for '{target}':\n")

    results = recommend_keywords(target, top_k=5)

    for kw, sim, stab, score in results:
        print(f"{kw:20} | sim={sim:.3f} | stab={stab:.3f} | score={score:.3f}")
