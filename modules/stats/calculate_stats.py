import os
import sys

sys.path.append("../../")

import numpy as np

from modules.utils.load_dataset import *
from modules.utils.stats_utils import *

results_path = "../../results"

class Statistics():
    def __init__(self):
        combined = load_combined_reviews()
        self.user_ids = np.unique(combined["user_id"])
        user2group = {user_id: group for user_id, group in combined[["user_id", "group"]].values}
        self.groups = [user2group[user_id] for user_id in self.user_ids]

        gmap2city = get_gmap_id_to_city()

        combined["review_length"] = combined["text"].apply(get_text_length)
        combined["year_month"] = combined["time"].apply(convert_to_year_month)
        combined["city"] = combined["gmap_id"].apply(lambda x: gmap2city[x])
        combined["sentiment"] = combined["text"].apply(lambda x: get_sentiment(x))
        combined["diversity"] = combined["text"].apply(lambda x: get_diversity(x))

        self.combined = combined

    def get_rating_mean(self):
        df = self.combined[["user_id", "group", "rating"]].groupby("user_id").agg({"group": "first", "rating": "mean"})
        df = df.reset_index()

        user2stats = {user:stat for user, stat in df[["user_id", "rating"]].values}
        return user2stats

    def get_rating_std(self):
        df = self.combined[["user_id", "group", "rating"]].groupby("user_id").agg({"group": "first", "rating": "std"})
        df = df.reset_index()

        user2stats = {user:stat for user, stat in df[["user_id", "rating"]].values}
        return user2stats

    def get_num_reviews(self):
        df = self.combined[["user_id", "group"]].groupby("user_id").agg(["first", "count"])
        df = df.reset_index()
        df.columns = ["user_id", "group", "num_reviews"]

        user2stats = {user:stat for user, stat in df[["user_id", "num_reviews"]].values}
        return user2stats

    def get_num_reviews_monthly_mean(self):
        df = self.combined[["user_id", "group", "year_month"]].groupby(["user_id", "year_month"]).agg(["first", "count"])
        df = df.reset_index()
        df.columns = ["user_id", "year_month", "group", "num_reviews"]

        df = df.groupby("user_id").agg({"group": "first", "num_reviews": "mean"})
        df = df.reset_index()

        user2stats = {user:stat for user, stat in df[["user_id", "num_reviews"]].values}
        return user2stats

    def get_num_reviews_monthly_std(self):
        df = self.combined[["user_id", "group", "year_month"]].groupby(["user_id", "year_month"]).agg(["first", "count"])
        df = df.reset_index()
        df.columns = ["user_id", "year_month", "group", "num_reviews"]

        df = df.groupby("user_id").agg({"group": "first", "num_reviews": "std"})
        df = df.reset_index().fillna(0.0)

        user2stats = {user:stat for user, stat in df[["user_id", "num_reviews"]].values}
        return user2stats

    def get_num_reviews_monthly_max(self):
        df = self.combined[["user_id", "group", "year_month"]].groupby(["user_id", "year_month"]).agg(["first", "count"])
        df = df.reset_index()
        df.columns = ["user_id", "year_month", "group", "num_reviews"]

        df = df.groupby("user_id").agg({"group": "first", "num_reviews": "max"})
        df = df.reset_index()

        user2stats = {user:stat for user, stat in df[["user_id", "num_reviews"]].values}
        return user2stats

    def get_review_length_mean(self):
        df = self.combined[["user_id", "group", "review_length"]].groupby("user_id").agg({"group": "first", "review_length": "mean"})
        df = df.reset_index()

        user2stats = {user:stat for user, stat in df[["user_id", "review_length"]].values}
        return user2stats

    def get_review_length_std(self):
        df = self.combined[["user_id", "group", "review_length"]].groupby("user_id").agg({"group": "first", "review_length": "std"})
        df = df.reset_index()

        user2stats = {user:stat for user, stat in df[["user_id", "review_length"]].values}
        return user2stats

    def get_num_cities(self):
        df = self.combined[["user_id", "group", "city"]]
        df = df.groupby("user_id").agg({"group": "first", "city": "nunique"})
        df = df.reset_index()
        df.columns = ["user_id", "group", "num_cities"]

        user2stats = {user:stat for user, stat in df[["user_id", "num_cities"]].values}
        return user2stats

    def get_sentiment_mean(self):
        df = self.combined[["user_id", "group", "sentiment"]]
        df = df.groupby("user_id").agg({"group": "first", "sentiment": "mean"})
        df = df.reset_index()
        df.columns = ["user_id", "group", "sentiment"]

        user2stats = {user:stat for user, stat in df[["user_id", "sentiment"]].values}
        return user2stats

    def get_sentiment_std(self):
        df = self.combined[["user_id", "group", "sentiment"]]
        df = df.groupby("user_id").agg({"group": "first", "sentiment": "std"})
        df = df.reset_index()
        df.columns = ["user_id", "group", "sentiment"]

        user2stats = {user:stat for user, stat in df[["user_id", "sentiment"]].values}
        return user2stats

    def get_activity_months(self):
        df = self.combined[["user_id", "group", "year_month"]]
        df = df.groupby("user_id").agg({"group": "first", "year_month": "nunique"})
        df = df.reset_index()
        df.columns = ["user_id", "group", "activity"]

        user2stats = {user:stat for user, stat in df[["user_id", "activity"]].values}
        return user2stats

    def get_activity_duration(self):
        unix_month = 2629743 * 1000
        df = self.combined[["user_id", "group", "time"]]
        df = df.groupby("user_id").agg({"group": "first", "time": ["max", "min"]})
        df = df.reset_index()
        df.columns = ["user_id", "group", "max", "min"]
        df["activity"] = (df["max"] - df["min"]) / unix_month

        user2stats = {user:stat for user, stat in df[["user_id", "activity"]].values}
        return user2stats

    def get_activity_rating_diff(self):
        df = self.combined[["user_id", "group", "time", "rating"]]
        df = df.sort_values(by="time")
        df = df[["user_id", "group", "rating", "time"]]
        df = df.groupby("user_id").agg({
            "group": "first",
            "rating": [lambda x: x.head(10).mean(), lambda x: x.tail(10).mean()]
        })

        df = df.reset_index()
        df.columns = ["user_id", "group", "first", "last"]
        df["activity"] = df["last"] - df["first"]

        user2stats = {user:stat for user, stat in df[["user_id", "activity"]].values}
        return user2stats

    def get_diversity_mean(self):
        df = self.combined[["user_id", "group", "diversity"]]
        df = df.groupby("user_id").agg({"group": "first", "diversity": "mean"})
        df = df.reset_index()
        df.columns = ["user_id", "group", "diversity"]

        user2stats = {user:stat for user, stat in df[["user_id", "diversity"]].values}
        return user2stats

    def get_diversity_std(self):
        df = self.combined[["user_id", "group", "diversity"]]
        df = df.groupby("user_id").agg({"group": "first", "diversity": "std"})
        df = df.reset_index()
        df.columns = ["user_id", "group", "diversity"]

        user2stats = {user:stat for user, stat in df[["user_id", "diversity"]].values}
        return user2stats

    def save_stats(self):
        with open(f"{results_path}/stats.txt", "r") as f:
            stats_list = f.read().split()

        results = {"user_id": list(self.user_ids), "group": list(self.groups)}
        for stats in stats_list:
            print(f"Processing {stats} statistics")
            method_name = f"get_{stats}"
            mapping = getattr(self, method_name)()
            results[stats] = [mapping[user_id] for user_id in self.user_ids]

        results = pd.DataFrame(results)
        print(results.head(5))

        results.to_csv(f"{results_path}/stats.csv", index=False)

if __name__ == "__main__":
    if not os.path.exists(f"{results_path}/stats.csv"):
        statistics = Statistics()
        statistics.save_stats()
