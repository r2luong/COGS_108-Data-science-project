import sys
import json

sys.path.append("../../")

import numpy as np

from modules.utils.load_dataset import *
from modules.utils.plot_utils import *
from modules.utils.stats_utils import *

results_path = "../../results"

class DistributionVisualizer():
    def __init__(self, stats):
        self.stats = stats
        self.num_bins = 100

    def plot_activity_months(self):
        filtered_stats = self.stats[self.stats["activity_months"] <= 50]

        config = {
            "stats": self.stats,
            "filtered_stats": filtered_stats,
            "name": "activity_months",
            "desc": "number of activity months",
            "bins": np.arange(50),
            "opt_desc": "excluded users whose number of activity months is larger than 50."
        }

        return config

    def plot_activity_duration(self):
        filtered_stats = self.stats[self.stats["activity_duration"] <= 100]

        config = {
            "stats": self.stats,
            "filtered_stats": filtered_stats,
            "name": "activity_duration",
            "desc": "length of activity duration (months)",
            "bins": np.linspace(0, 100, self.num_bins),
            "opt_desc": "excluded users whose number of activity duration (months) is longer than 100."
        }

        return config

    def plot_activity_rating_diff(self):
        config = {
            "stats": self.stats,
            "filtered_stats": self.stats,
            "name": "activity_rating_diff",
            "desc": "difference of ratings between first and last 10 reviews (last - first)",
            "bins": np.linspace(-4, 4, self.num_bins),
            "opt_desc": None
        }

        return config

    def plot_diversity_mean(self):
        config = {
            "stats": self.stats,
            "filtered_stats": self.stats,
            "name": "diversity_mean",
            "desc": "mean of diversity of user reviews",
            "bins": np.linspace(1.0, 1.5, self.num_bins),
            "opt_desc": None
        }

        return config

    def plot_diversity_std(self):
        filtered_stats = self.stats[self.stats["diversity_std"] <= 0.5]

        config = {
            "stats": self.stats,
            "filtered_stats": filtered_stats,
            "name": "diversity_std",
            "desc": "std of diversity of user reviews",
            "bins": np.linspace(0, 0.5, 50),
            "opt_desc": "excluded users whose std of diversity of review words is larger than 0.5."
        }

        return config

    def plot_num_cities(self):
        filtered_stats = self.stats[self.stats["num_cities"] <= 50]

        config = {
            "stats": self.stats,
            "filtered_stats": filtered_stats,
            "name": "num_cities",
            "desc": "number of cities of user ratings",
            "bins": np.arange(50),
            "opt_desc": "excluded users whose number of cities is larger than 50."
        }

        return config

    def plot_num_reviews(self):
        filtered_stats = self.stats[self.stats["num_reviews"] <= 100]

        config = {
            "stats": self.stats,
            "filtered_stats": filtered_stats,
            "name": "num_reviews",
            "desc": "number of user ratings",
            "bins": np.arange(10, 100),
            "opt_desc": "excluded users with more than 100 reviews"
        }

        return config

    def plot_num_reviews_monthly_mean(self):
        filtered_stats = self.stats[self.stats["num_reviews_monthly_mean"] <= 10]

        config = {
            "stats": self.stats,
            "filtered_stats": filtered_stats,
            "name": "num_reviews_monthly_mean",
            "desc": "mean of numbers of user ratings per month",
            "bins": np.linspace(0, 10, self.num_bins),
            "opt_desc": "excluded users whose mean of reviews per month is larger than 10."
        }

        return config

    def plot_num_reviews_monthly_std(self):
        filtered_stats = self.stats[self.stats["num_reviews_monthly_std"] <= 10]

        config = {
            "stats": self.stats,
            "filtered_stats": filtered_stats,
            "name": "num_reviews_monthly_std",
            "desc": "std of numbers of user ratings per month",
            "bins": np.linspace(0, 10, self.num_bins),
            "opt_desc": "excluded users whose std of reviews per month is larger than 10."
        }

        return config

    def plot_num_reviews_monthly_max(self):
        filtered_stats = self.stats[self.stats["num_reviews_monthly_max"] <= 30]

        config = {
            "stats": self.stats,
            "filtered_stats": filtered_stats,
            "name": "num_reviews_monthly_max",
            "desc": "max of numbers of user ratings per month",
            "bins": np.arange(0, 30),
            "opt_desc": "excluded users whose max of reviews per month is larger than 30."
        }

        return config


    def plot_rating_mean(self):
        config = {
            "stats": self.stats,
            "filtered_stats": self.stats,
            "name": "rating_mean",
            "desc": "mean of user ratings",
            "bins": np.linspace(1.0, 5.0, self.num_bins),
        }

        return config

    def plot_rating_std(self):
        config = {
            "stats": self.stats,
            "filtered_stats": self.stats,
            "name": "rating_std",
            "desc": "std of user ratings",
            "bins": np.linspace(0, 2.5, self.num_bins),
        }

        return config

    def plot_review_length_mean(self):
        filtered_stats = self.stats[self.stats["review_length_mean"] <= 100]

        config = {
            "stats": self.stats,
            "filtered_stats": filtered_stats,
            "name": "review_length_mean",
            "desc": "mean of review lengths of user ratings",
            "bins": np.linspace(0, 100, self.num_bins),
            "opt_desc": "excluded users whose mean of review lengths is larger than 100."
        }

        return config

    def plot_review_length_std(self):
        filtered_stats = self.stats[self.stats["review_length_std"] <= 100]

        config = {
            "stats": self.stats,
            "filtered_stats": filtered_stats,
            "name": "review_length_std",
            "desc": "std of review lengths of user ratings",
            "bins": np.linspace(0, 100, self.num_bins),
            "opt_desc": "excluded users whose std of review lengths is larger than 100."
        }

        return config

    def plot_sentiment_mean(self):
        config = {
            "stats": self.stats,
            "filtered_stats": self.stats,
            "name": "sentiment_mean",
            "desc": "mean of sentiment of user reviews",
            "bins": np.linspace(-1, 1, self.num_bins),
            "opt_desc": None
        }

        return config

    def plot_sentiment_std(self):
        config = {
            "stats": self.stats,
            "filtered_stats": self.stats,
            "name": "sentiment_std",
            "desc": "std of sentiment of user reviews",
            "bins": np.linspace(0, 1.0, self.num_bins),
            "opt_desc": None
        }

        return config

    def plot_stats(self, name, transform=None, show=False):
        method_name = f"plot_{name}"
        config = getattr(self, method_name)()

        if transform is not None:
            config["transform"] = transform
            plot_transform(config, show)

        else:
            plot_distribution(config, show)

    def plot_all_stats(self):
        with open(f"{results_path}/stats.txt", "r") as f:
            stats_list = f.read().split()

        for stats in stats_list:
            print(f"Plotting {stats} statistics")
            self.plot_stats(stats)

    def plot_all_stats_transform(self):
        with open(f"{results_path}/stats.txt", "r") as f:
            stats_list = f.read().split()

        with open(f"{results_path}/manual_distribution.json", "r") as f:
            dist = json.load(f)

        for stats in stats_list:
            stat_dist = dist[stats]
            if stat_dist["normal"] == 0:
                for transform in ["outlier", "log", "reciprocal", "sqrt", "cbrt"]:
                    print(f"Plotting {stats} statistics with {transform} transform")
                    self.plot_stats(stats, transform=transform)

if __name__ == "__main__":
    stats = load_stats()
    visualizer = DistributionVisualizer(stats)
    visualizer.plot_all_stats()
    visualizer.plot_all_stats_transform()