import os
import json
import gzip
from functools import partial

import numpy as np
import pandas as pd
import tqdm

raw_path = "../../data/00-raw"
interim_path = "../../data/01-interim"

review_path = "../../data/00-raw/review-California.json.gz"
review_keys = ["gmap_id", "user_id", "name", "time", "rating", "text"]

total_reviews = 70529977

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)

def filter_by_gmap_id(data, gmap_ids):
    gmap_id = data.get("gmap_id", None)
    if gmap_id is None:
        return False

    return gmap_id in gmap_ids

def filter_raw_review_data(filters):
    reviews = []
    for review in tqdm.tqdm(parse(review_path), total=total_reviews):
        if all([f(data=review) for f in filters]):
            review = {key: review.get(key, None) for key in review_keys}
            reviews.append(review)

    print(f"We obtained total of {len(reviews)} reviews after filtering")

    df = pd.DataFrame(reviews)
    print(df.head())

    print(f"The number of reviews before dropping reviews with missing entry is {df.shape[0]}")
    df = df.dropna().reset_index(drop=True)
    print(f"The number of reviews after dropping reviews with missing entry is {df.shape[0]}")
    print(df.isnull().sum())

    df.to_csv(f"{interim_path}/reviews.csv", index=False)

def extract_user_ids(reviews, min_num_reviews):
    user_ids = reviews["user_id"].values
    unique, counts = np.unique(np.array(user_ids), return_counts=True)
    users = pd.DataFrame({"user_id": unique, "num_reviews": counts})

    users = users[users["num_reviews"] >= min_num_reviews].reset_index(drop=True)
    print(f"We found {users.shape[0]} users in total.")
    users.to_csv(f"{interim_path}/users.csv", index=False)

def filter_by_user_ids(reviews, user_ids):
    reviews = reviews[reviews["user_id"].isin(user_ids)]
    reviews = reviews.dropna().reset_index(drop=True)
    print(f"We found {reviews.shape[0]} reviews assocaited with users we extracted.")
    reviews.to_csv(f"{interim_path}/user_reviews.csv", index=False)

if __name__ == "__main__":
    gmap_ids = set(pd.read_csv(f"{interim_path}/restaurants.csv")["gmap_id"].values)

    if not os.path.exists(f"{interim_path}/reviews.csv"):
        gmap_id_filter = partial(filter_by_gmap_id, gmap_ids=gmap_ids)
        filter_raw_review_data([gmap_id_filter])

    if not os.path.exists(f"{interim_path}/users.csv"):
        print("Start processing user data")
        reviews = pd.read_csv(f"{interim_path}/reviews.csv")
        min_num_reviews = 10
        extract_user_ids(reviews, min_num_reviews)

    if not os.path.exists(f"{interim_path}/user_reviews.csv"):
        print("Start filtering review data")
        reviews = pd.read_csv(f"{interim_path}/reviews.csv")
        user_ids = pd.read_csv(f"{interim_path}/users.csv")["user_id"].values
        filter_by_user_ids(reviews, user_ids)