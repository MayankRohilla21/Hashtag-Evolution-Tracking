import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import itertools

# Load cleaned dataset
df = pd.read_csv("reddit_politics_cleaned.csv")

# Use the CLEAN tokens generated earlier
df['clean_tokens'] = df['clean_tokens'].apply(eval)  # convert string → list

# Count top words from clean tokens
all_tokens = [tok for ts in df['clean_tokens'] for tok in ts]
common = [w for w, c in Counter(all_tokens).most_common(50)]

# Build co-occurrence graph
G = nx.Graph()

for tokens in df['clean_tokens']:
    filtered = [t for t in tokens if t in common]  # only top words
    for a, b in itertools.combinations(set(filtered), 2):
        if G.has_edge(a, b):
            G[a][b]['weight'] += 1
        else:
            G.add_edge(a, b, weight=1)

# Draw graph
plt.figure(figsize=(13,10))
pos = nx.spring_layout(G, k=0.5)
weights = [G[u][v]['weight'] for u, v in G.edges()]

nx.draw_networkx(
    G, pos,
    with_labels=True,
    width=weights,
    node_color='lightblue',
    edge_color='gray',
    font_size=10
)

plt.title("Keyword Co-occurrence Graph (Cleaned)")
plt.show()
