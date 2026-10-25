import os
import sys
import re

sys.path.append("../../")

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.corpus import stopwords

from modules.utils.load_dataset import *

palette = {"critical": "#a02121", "neutral": "#54d582", "generous": "#5497d5"}
order = ["critical", "neutral", "generous"]
results_path = "../../results"

def plot_wordcloude(texts, group, show):
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(texts)

    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for {group} users")
    plt.tight_layout()

    if show:
        plt.show()
    else:
        os.makedirs(f"{results_path}/texts", exist_ok=True)
        plt.savefig(f"{results_path}/texts/wordcloude_{group}.png")
        plt.close()

def plot_wordclouds(combined, show=False):
    groups = np.unique(combined["group"].values)

    for group in groups:
        texts = " ".join(combined[combined["group"] == group]["text"].values)
        plot_wordcloude(texts, group, show)

def find_most_common_words(texts, num_words):
    texts = re.sub("[^0-9A-Za-z]", " ", texts).lower()
    wordlist = set(stopwords.words("english"))
    tokens = [t for t in texts.split() if t not in wordlist]

    unique, counts = np.unique(tokens, return_counts=True)
    indices = np.argsort(counts)[::-1][:num_words]
    return unique[indices]

def create_count_dict(texts, num_reviews):
    texts = re.sub("[^0-9A-Za-z]", " ", texts).lower()
    wordlist = set(stopwords.words("english"))
    tokens = [t for t in texts.split() if t not in wordlist]

    unique, counts = np.unique(tokens, return_counts=True)

    word2count = {word:count / num_reviews for word, count in zip(unique, counts)}
    return word2count

def plot_most_common_words_by_group(combined, show=False):
    groups = np.unique(combined["group"].values)

    num_words = 20
    most_common = find_most_common_words(" ".join(combined["text"].values), num_words)

    counts = []
    for group in groups:
        group_reviews = combined[combined["group"] == group]
        texts = " ".join(group_reviews["text"].values)
        count_dict = create_count_dict(texts, group_reviews.shape[0])
        counts += [(word, count_dict[word], group) for word in most_common]

    df = pd.DataFrame(counts, columns=["word", "count", "group"])

    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))
    sns.barplot(df, x="count", y="word", hue="group", palette=palette, order=most_common, hue_order=order, alpha=0.8)

    plt.title(f"Most commond words by user group")
    plt.xlabel("Frequency divided by total number of reviews per group")
    plt.ylabel("Most common words")
    plt.tight_layout()

    if show:
        plt.show()
    else:
        os.makedirs(f"{results_path}/texts", exist_ok=True)
        plt.savefig(f"{results_path}/texts/most_common.png")
        plt.close()

if __name__ == "__main__":
    combined = load_combined_reviews()

    plot_wordclouds(combined)
    plot_most_common_words_by_group(combined)
