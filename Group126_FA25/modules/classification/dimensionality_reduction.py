import sys
import os

sys.path.append("../../")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from modules.utils.load_dataset import *

palette = {"critical": "#a02121", "neutral": "#54d582", "generous": "#5497d5"}
order = ["critical", "neutral", "generous"]

def visualize_top_three_components(stats, n_components, model, name, show=False):
    print(f"Analyzing dimensionality reduction with {name}")
    vars = list(stats.columns)[2:]
    values = stats[vars].values

    component_columns = [f"component_{i}" for i in range(n_components)]
    components = pd.DataFrame(model.fit_transform(values), columns=component_columns)
    users = stats[["user_id", "group"]]

    results = pd.concat([users, components], axis=1)
    _, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 10))

    sns.set_style("whitegrid")
    plt.suptitle(f"Distributions of users based on top {n_components} components of {name}.")

    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, pair in enumerate(pairs):
        plot_samples(axes[i], results, pair)

    plt.tight_layout()

    if show:
         plt.show()
    else:
        os.makedirs(f"{results_path}/dimension", exist_ok=True)
        plt.savefig(f"{results_path}/dimension/{name}_top_{n_components}.png")
        plt.close()

        results.to_csv(f"{results_path}/dimension/{name}_top_{n_components}.csv", index=False)

def plot_samples(ax, results, pair):
        ax.set_title(f"Distribution of users based on component {pair[0]} and {pair[1]}")
        ax.set_xlabel(f"Component {pair[0]}")
        ax.set_ylabel(f"Component {pair[1]}")

        sns.scatterplot(results, x=f"component_{pair[0]}", y=f"component_{pair[1]}",
                        hue="group", palette=palette, ax=ax)

if __name__ == "__main__":
    stats = load_stats()

    n_components = 3
    pca = PCA(n_components=n_components)
    visualize_top_three_components(stats, n_components, pca, "PCA")

    tsne = TSNE(n_components=n_components)
    visualize_top_three_components(stats, n_components, tsne, "t-SNE")
