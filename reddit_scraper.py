import praw
import pandas as pd
from tqdm import tqdm

reddit = praw.Reddit(
    client_id="1mfqbdi6fmvCQCZbY1tlWg",
    client_secret="Xqivpjn6PW-hNz3TtwaQrrUuCOlBbQ",
    user_agent="hashtag tracker by /u/Mountain_Tonight1228"
)

subreddit = reddit.subreddit("politics")

data = []
for submission in tqdm(subreddit.search("gaslighting", limit=500)):
    data.append({
        "title": submission.title,
        "selftext": submission.selftext,
        "created_utc": submission.created_utc,
        "score": submission.score,
        "num_comments": submission.num_comments,
        "url": submission.url
    })

df = pd.DataFrame(data)
df.to_csv("reddit_politics_posts.csv", index=False)
print("Data saved to reddit_politics_posts.csv")
