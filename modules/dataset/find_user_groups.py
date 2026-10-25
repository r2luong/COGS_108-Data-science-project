import os

import numpy as np
import pandas as pd

interim_path = "../../data/01-interim"
processed_path = "../../data/02-processed"

np.random.seed(0)

def filter_reviews_by_grouped_restaurants(reviews, name):
    restaurants = pd.read_csv(f"{interim_path}/{name}_rated_restaurants.csv")
    gmap_ids = restaurants["gmap_id"].values
    reviews = reviews[reviews["gmap_id"].isin(gmap_ids)]

    print(f"We found {reviews.shape[0]} reviews for {name} rated restaurants")
    reviews = reviews.reset_index(drop=True)
    reviews.to_csv(f"{interim_path}/{name}_rated_restaurants_reviews.csv", index=False)

def find_critical_users(min_num_reviews):
    restaurants_reviews = pd.read_csv(f"{interim_path}/highly_rated_restaurants_reviews.csv")
    harsh_reviews = restaurants_reviews[restaurants_reviews["rating"].isin([1.0, 2.0])]

    user_ids = harsh_reviews["user_id"].values
    unique, counts = np.unique(np.array(user_ids), return_counts=True)

    critical_users = unique[counts >= min_num_reviews]
    print(f"We found {critical_users.shape[0]} critical users")

    with open(f"{interim_path}/critical_users.txt", "w") as f:
        f.write("\n".join(critical_users))

def find_generous_users(min_num_reviews):
    restaurants_reviews = pd.read_csv(f"{interim_path}/harshly_rated_restaurants_reviews.csv")
    generous_reviews = restaurants_reviews[restaurants_reviews["rating"].isin([5.0])]

    user_ids = generous_reviews["user_id"].values
    unique, counts = np.unique(np.array(user_ids), return_counts=True)

    generous_users = unique[counts >= min_num_reviews]
    print(f"We found {generous_users.shape[0]} generous users.")

    with open(f"{interim_path}/generous_users.txt", "w") as f:
        f.write("\n".join(generous_users))

def downsample_users():
    with open(f"{interim_path}/critical_users.txt", "r") as f:
        critical_users = f.read().split()

    with open(f"{interim_path}/generous_users.txt", "r") as f:
        generous_users = f.read().split()

    all_users = pd.read_csv(f"{interim_path}/users.csv")["user_id"].values
    neutral_users = [user for user in all_users if user not in list(critical_users) + list(generous_users)]

    intersection = [u for u in critical_users if u in generous_users]
    critical_users = [u for u in critical_users if u not in intersection]
    generous_users = [u for u in generous_users if u not in intersection]

    user_size = min(len(critical_users), len(generous_users))
    print(f"We have {user_size} users for each group.")

    critical_users = np.random.choice(critical_users, size=user_size, replace=False)
    generous_users = np.random.choice(generous_users, size=user_size, replace=False)
    neutral_users = np.random.choice(neutral_users, size=user_size, replace=False)

    assert len(critical_users) == len(generous_users) ==  len(neutral_users)
    intersection = [u for u in critical_users if u in generous_users]
    assert len(intersection) == 0

    reviews = pd.read_csv(f"{interim_path}/user_reviews.csv")

    critical_user_reviews = reviews[reviews["user_id"].isin(critical_users)]
    print(f"We found {critical_user_reviews.shape[0]} reviews by critical users.")
    critical_user_reviews.to_csv(f"{processed_path}/critical_user_reviews.csv", index=False)

    generous_user_reviews = reviews[reviews["user_id"].isin(generous_users)]
    print(f"We found {generous_user_reviews.shape[0]} reviews by generous users.")
    generous_user_reviews.to_csv(f"{processed_path}/generous_user_reviews.csv", index=False)

    neutral_user_reviews = reviews[reviews["user_id"].isin(neutral_users)]
    print(f"We found {neutral_user_reviews.shape[0]} reviews by neutral users.")
    neutral_user_reviews.to_csv(f"{processed_path}/neutral_user_reviews.csv", index=False)


if __name__ == "__main__":
    names = ["harshly", "highly"]
    for name in names:
        if not os.path.exists(f"{interim_path}/{name}_rated_restaurants_reviews.csv"):
            reviews = pd.read_csv(f"{interim_path}/user_reviews.csv")
            filter_reviews_by_grouped_restaurants(reviews, name)

    min_num_reviews = 2
    if not os.path.exists(f"{interim_path}/critical_users.txt"):
        find_critical_users(min_num_reviews)

    if not os.path.exists(f"{interim_path}/generous_users.txt"):
        find_generous_users(min_num_reviews)

    if not os.path.exists(f"{processed_path}/neutral_user_reviews.csv"):
        downsample_users()
