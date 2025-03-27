import os
import praw
import pandas as pd
import datetime as dt
from tqdm import tqdm
import time
import json


def reddit_connection():
    personal_use_script = os.environ["REDDIT_PERSONAL_USE_SCRIPT_14_CHARS"]
    client_secret = os.environ["REDDIT_SECRET_KEY_27_CHARS"]
    user_agent = os.environ["REDDIT_APP_NAME"]
    username = os.environ["REDDIT_USER_NAME"]
    password = os.environ["REDDIT_LOGIN_PASSWORD"]

    reddit = praw.Reddit(client_id=personal_use_script, \
                         client_secret=client_secret, \
                         user_agent=user_agent, \
                         username=username, \
                         password='')
    return reddit


def build_dataset(reddit, search_words='ConspiracyTheory', items_limit=2000):
    
    # Collect reddit posts
    subreddit = reddit.subreddit(search_words)
    new_subreddit = subreddit.new(limit=items_limit)

    fine_tune_data = []
    
    for submission in tqdm(new_subreddit):
        # Skip posts without body
        if not submission.selftext:
            continue

        submission.comments.replace_more(limit=0)
        top_comments = submission.comments.list()
        
        if len(top_comments) == 0:
            continue

        # Pick the top comment (highest upvoted)
        top_comment = sorted(top_comments, key=lambda x: x.score, reverse=True)[0]
        
        fine_tune_data.append({
            "messages": [
                {"role": "user", "content": submission.title + "\n\n" + submission.selftext},
                {"role": "assistant", "content": top_comment.body}
            ]
        })

    return fine_tune_data

def save_dataset(fine_tune_data):
    with open("conspiracy_finetune_dataset.jsonl", "w") as f:
        for example in fine_tune_data:
            json.dump(example, f)
            f.write("\n")


if __name__ == "__main__": 
	reddit = reddit_connection()
	fine_tune_data = build_dataset(reddit)
	save_dataset(fine_tune_data)
