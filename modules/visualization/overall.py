import os
import sys

sys.path.append("../../")

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from modules.utils.load_dataset import *

palette = {"critical": "#a02121", "neutral": "#54d582", "generous": "#5497d5"}
order = ["critical", "neutral", "generous"]
results_path = "../../results"

def plot_overall_rating_bar(combined, show=False):
    df = combined["rating"].value_counts()
    df = df.reset_index()

    sns.set_style("whitegrid")
    plt.title("Distribution of ratings of all reviews")
    plt.xlabel("Rating")
    plt.ylabel("The number of reviews")

    sns.barplot(df, x="rating", y="count", order=[1.0, 2.0, 3.0, 4.0, 5.0], alpha=0.8)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        os.makedirs(f"{results_path}/overall", exist_ok=True)
        plt.savefig(f"{results_path}/overall/overall_rating.png")
        plt.close()

def plot_overall_num_reviews_bar(combined, show=False):
    df = combined["group"].value_counts()
    df = df.reset_index()

    sns.set_style("whitegrid")
    plt.title("Total number of reviews by user groups")
    plt.xlabel("User groups")
    plt.ylabel("The number of reviews")

    sns.barplot(df, x="group", y="count", hue="group", order=order, palette=palette)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        os.makedirs(f"{results_path}/overall", exist_ok=True)
        plt.savefig(f"{results_path}/overall/overall_num_reviews.png")
        plt.close()

def plot_rating_all_five_users(combined, show=False):
    df = combined[["user_id", "group", "rating"]].groupby("user_id").agg({"group": "first", "rating": "mean"})
    df = df.reset_index()

    ratios = []
    avg_five_users = df[df["rating"] == 5.0]
    for name in order:
        ratio = np.sum(avg_five_users["group"] == name) / np.sum(df["group"] == name)
        ratios.append((name, ratio))

    df = pd.DataFrame(ratios, columns=["group", "ratio"])

    sns.set_style("whitegrid")
    plt.title("Proportion of users who rate all restaurants 5 by user group")
    plt.xlabel("User groups")
    plt.ylabel("Proportion of all five rating users")

    sns.barplot(df, x="group", y="ratio", hue="group", order=order, palette=palette)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        os.makedirs(f"{results_path}/overall", exist_ok=True)
        plt.savefig(f"{results_path}/overall/all_five.png")
        plt.close()

if __name__ == "__main__":
    combined = load_combined_reviews()

    plot_overall_rating_bar(combined)
    plot_overall_num_reviews_bar(combined)
    plot_rating_all_five_users(combined)