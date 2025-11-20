**Hashtag Evolution Tracker (Reddit-Based)**

This project analyzes how the meaning of a hashtag evolves over time on Reddit and recommends more stable, semantically similar alternatives. Marketers, researchers, and trend analysts can use this tool to avoid hashtags whose meaning becomes volatile due to political events, cultural shifts, or online discourse.

This repository demonstrates the full pipeline using one example hashtag: gaslighting inside the politics subreddit.
However, the workflow is fully generalizable to any keyword or domain simply by updating the scraper input.

Features

* Scrapes Reddit posts containing target keywords

* Cleans and preprocesses text + extracts hashtags

* Computes sentiment and usage trends over time

* Builds co-occurrence keyword graphs

* Extracts top keywords using TF-IDF

* Finds semantically similar keywords using Sentence-BERT

* Measures semantic shift across years

* Ranks stable alternatives for high-risk hashtags

python reddit_scraper.py
python text_processing.py
python analysis_and_viz.py
python co_occurrence_graph.py
python semantic_neighbors.py
python 06_semantic.py
python rankings.py


Installation of requirements

pip install praw pandas numpy nltk textblob tqdm matplotlib networkx sentence-transformers scikit-learn

NLTK data
import nltk
nltk.download("punkt")
nltk.download("stopwords")

