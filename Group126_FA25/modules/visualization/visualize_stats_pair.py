import sys

sys.path.append("../../")

import numpy as np

from modules.utils.load_dataset import *
from modules.utils.plot_utils import *
from modules.utils.stats_utils import *

class PairVisualizer():
    def __init__(self, stats):
        self.stats = stats

    def plot_rating_mean_review_length_mean(self):
        filtered_stats = self.stats[self.stats["review_length_mean"] <= 100]

        config = {
            "stats": self.stats,
            "filtered_stats": filtered_stats,
            "name_x": "rating_mean",
            "name_y": "review_length_mean",
            "desc_x": "mean of ratings",
            "desc_y": "mean of review lengths",
            "ratio_range": [0, 20],
            "bins": np.linspace(0, 20, 50),
            "opt_ratio_desc": "excluded users whose ratio is larger than 2.0.",
            "opt_scatter_desc": "excluded users whose mean of review length is larger than 100."
        }

        return config

    def plot_num_reviews_num_cities(self):
        filtered_stats = self.stats[np.logical_and(self.stats["num_cities"] <= 50, self.stats["num_reviews"] <= 200)]

        config = {
            "stats": self.stats,
            "filtered_stats": filtered_stats,
            "name_x": "num_reviews",
            "name_y": "num_cities",
            "desc_x": "number of user reviews",
            "desc_y": "number of cities of restaurants a user rated",
            "ratio_range": [0, 1.0],
            "bins": np.linspace(0, 1.0, 50),
            "opt_ratio_desc": None,
            "opt_scatter_desc": "excluded users whose number of cities is larger than 50 or number of reviews is larger than 200."
        }

        return config

    def plot_rating_mean_sentiment_mean(self):
        filtered_stats = self.stats

        config = {
            "stats": self.stats,
            "filtered_stats": filtered_stats,
            "name_x": "rating_mean",
            "name_y": "sentiment_mean",
            "desc_x": "mean of user ratings",
            "desc_y": "mean of review sentiment",
            "ratio_range": [-0.2, 0.2],
            "bins": np.linspace(-0.2, 0.2, 50),
            "opt_ratio_desc": "excluded users whose ratio is outside of [-0.2, 0.2]",
            "opt_scatter_desc": None
        }

        return config

    def plot_rating_std_sentiment_std(self):
        filtered_stats = self.stats

        config = {
            "stats": self.stats,
            "filtered_stats": filtered_stats,
            "name_x": "rating_std",
            "name_y": "sentiment_std",
            "desc_x": "std of user ratings",
            "desc_y": "std of review sentiment",
            "ratio_range": [0, 1.0],
            "bins": np.linspace(0, 1.0, 50),
            "opt_ratio_desc": "excluded users whose ratio is outside of [0, 1.0]",
            "opt_scatter_desc": None
        }

        return config

    def plot_activity_months_activity_duration(self):
        filtered_stats = self.stats[np.logical_and(self.stats["activity_months"] <= 50, self.stats["activity_duration"] <= 150)]

        config = {
            "stats": self.stats,
            "filtered_stats": filtered_stats,
            "name_x": "activity_months",
            "name_y": "activity_duration",
            "desc_x": "number of active months of user",
            "desc_y": "length of active duration of user (months)",
            "ratio_range": [0, 10],
            "bins": np.linspace(0, 10.0, 50),
            "opt_ratio_desc": None,
            "opt_scatter_desc": "excluded users whose number of activity month is larger than 50 or activei duration of user is larger than 150."
        }

        return config

    def plot_stats(self, name_x, name_y, show=False):
        method_name = f"plot_{name_x}_{name_y}"
        config = getattr(self, method_name)()
        plot_pair(config, show)

    def plot_all_stats(self):
        with open(f"{results_path}/pairs.txt", "r") as f:
            stats_list = [p.split(",") for p in f.read().split()]

        for stats in stats_list:
            print(f"Plotting {stats[0]} vs {stats[1]} statistics")
            self.plot_stats(*stats)


if __name__ == "__main__":
    stats = load_stats()
    visualizer = PairVisualizer(stats)
    visualizer.plot_all_stats()