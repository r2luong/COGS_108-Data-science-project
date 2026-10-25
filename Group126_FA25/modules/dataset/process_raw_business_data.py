import os
import json
import gzip
from functools import partial

import numpy as np
import pandas as pd

raw_path = "../../data/00-raw"
interim_path = "../../data/01-interim"

meta_path = "../../data/00-raw/meta-California.json.gz"
meta_keys = ["gmap_id", "name", "latitude", "longitude", "category", "avg_rating", "num_of_reviews", "address"]

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)

def analyze_raw_categories():
    categories = []
    for business in parse(meta_path):
        if business["category"] is not None:
            categories += business["category"]

    categories = np.array(categories)
    unique, counts = np.unique(categories, return_counts=True)
    print(f"The number of unique categories is {unique.shape[0]}")

    top_categories = unique[np.argsort(counts)[::-1][:5]]
    print(f"Top 5 frequent categories are {' / '.join(top_categories)}")

    restaurant_categories = [category for category in unique if "restaurant" in category.lower()]
    print(f"The number of categories containting 'restaurant' substring is {len(restaurant_categories)}")

    with open(f"{interim_path}/restaurant_categories.txt", "w") as f:
        f.write("\n".join(restaurant_categories))

def filter_by_category(data, categories):
    category = data.get("category", None)
    if category is None:
        return False

    return len(set(category) & categories) != 0

def filter_by_num_reviews(data, min_num_reviews):
    return data["num_of_reviews"] >= min_num_reviews

def filter_by_avg_rating(data, rating_range):
    avg_rating = data["avg_rating"]
    return (rating_range[0] < avg_rating) & (avg_rating <= rating_range[1])

def filter_raw_business_data(filters):
    businesses = []
    for business in parse(meta_path):
        if all([f(data=business) for f in filters]):
            business = {key: business.get(key, None) for key in meta_keys}
            businesses.append(business)

    print(f"We obtained total of {len(businesses)} after filtering")

    df = pd.DataFrame(businesses)
    print(df.head())

    df = df.dropna()
    print(f"The number of restaurants after dropping business with missing entry is {df.shape[0]}")
    df.to_csv(f"{interim_path}/restaurants.csv", index=False)

def filter_business_data(restaurants, filters, name):
    print(restaurants.head())

    for f in filters:
        restaurants = restaurants[f(restaurants)]

    print(f"There are {restaurants.shape[0]} {name} after filering.")
    restaurants = restaurants.reset_index(drop=True)
    restaurants.to_csv(f"{interim_path}/{name}.csv", index=False)

if __name__ == "__main__":
    if not os.path.exists(f"{interim_path}/restaurant_categories.txt"):
        analyze_raw_categories()

    if not os.path.exists(f"{interim_path}/restaurants.csv"):
        with open(f"{interim_path}/restaurant_categories.txt", "r") as f:
            restaurant_categories = set(f.read().split("\n"))

        min_num_reviews = 100
        restaurant_filter = partial(filter_by_category, categories=restaurant_categories)
        num_reviews_filter = partial(filter_by_num_reviews, min_num_reviews=min_num_reviews)

        filter_raw_business_data([restaurant_filter, num_reviews_filter])

    restaurants = pd.read_csv(f"{interim_path}/restaurants.csv")

    names = ["harshly", "highly"]
    rating_ranges = [[0, 3.5], [4.5, 5.0]]

    assert len(names) == len(rating_ranges)

    for name, rating_range in zip(names, rating_ranges):
        if not os.path.exists(f"{interim_path}/{name}_rated_restaurants.csv"):
            filter = partial(filter_by_avg_rating, rating_range=rating_range)
            filter_business_data(restaurants, filters=[filter], name=f"{name}_rated_restaurants")