import re
import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob

nltk.download('punkt')
nltk.download('stopwords')

STOP = set(stopwords.words('english'))

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove newlines
    text = re.sub(r'[\r\n]+', ' ', text)

    # Remove hashtags from text but keep them separately
    # (hashtags will still be extracted before removal)
    text = re.sub(r'#\w+', '', text)

    # Keep only alphanumeric text (after removing hashtags)
    text = re.sub(r'[^a-z0-9 ]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df = pd.read_csv("reddit_politics_posts.csv")

# Combine title + selftext and clean it
df['combined'] = (df['title'].fillna('') + ' ' + df['selftext'].fillna('')).apply(clean_text)

# Extract hashtags BEFORE removing them
df['hashtags'] = (df['title'].fillna('') + ' ' + df['selftext'].fillna('')).str.findall(r'#\w+')


def tokenize(text):
    return [
        tok for tok in text.split()
        if tok not in STOP and len(tok) > 2  # remove stopwords + small junk tokens
    ]

df['clean_tokens'] = df['combined'].apply(tokenize)


all_tokens = []
for toks in df['clean_tokens']:
    all_tokens.extend(toks)

common = Counter(all_tokens).most_common(50)
print("Top tokens:", common[:20])


df['sentiment'] = df['combined'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['date'] = pd.to_datetime(df['created_utc'], unit='s')

# Save cleaned dataset
df.to_csv("reddit_politics_cleaned.csv", index=False)
print("Saved reddit_politics_cleaned.csv")
