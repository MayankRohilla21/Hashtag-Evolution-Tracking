# analysis_and_viz.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("reddit_politics_cleaned.csv", parse_dates=['date'])
df['month'] = df['date'].dt.to_period('M')
trend = df.groupby('month').size()
trend.index = trend.index.to_timestamp()
trend.plot(title='Post count per month', marker='o')
plt.xlabel('Month'); plt.ylabel('Number of posts')
plt.tight_layout()
plt.savefig('trendpolitics.png')
plt.show()

# sentiment over time
sent = df.groupby('month')['sentiment'].mean()
sent.index = sent.index.to_timestamp()
sent.plot(title='Avg sentiment per month', marker='o')
plt.tight_layout()
plt.savefig('sentiment_trendpolitics.png')
plt.show()
print("Saved trend.png and sentiment_trend.png")