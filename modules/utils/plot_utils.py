import os

from functools import partial

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
import scipy

palette = {"critical": "#a02121", "neutral": "#54d582", "generous": "#5497d5"}
order = ["critical", "neutral", "generous"]
results_path = "../../results"

def plot_distribution(config, show=False):
    file_name = f"{results_path}/distribution/{config["name"]}.png"
    if not show and os.path.exists(file_name):
        return

    fig = plt.figure(figsize=(25, 12.5))
    gs = gridspec.GridSpec(2, 4)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_bar(ax1, config["stats"], config["name"], config["desc"])

    ax2 = fig.add_subplot(gs[0, 1])
    plot_qq(ax2, config["stats"], config["name"], config["desc"])

    ax3 = fig.add_subplot(gs[1, 0])
    plot_box(ax3, config["filtered_stats"], config["name"], config["desc"],
            opt_desc=config.get("opt_desc", None))

    ax4 = fig.add_subplot(gs[1, 1])
    plot_violin(ax4, config["filtered_stats"], config["name"], config["desc"],
            opt_desc=config.get("opt_desc", None))

    ax5 = fig.add_subplot(gs[:, 2:])
    plot_hist(ax5, config["filtered_stats"], config["name"], config["desc"],
            bins=config["bins"], opt_desc=config.get("opt_desc", None))

    plt.tight_layout()

    if show:
        plt.show()
    else:
        os.makedirs(f"{results_path}/distribution", exist_ok=True)
        plt.savefig(file_name)
        plt.close()

def apply_transform(stats, transform):
    if transform == "log":
        stats = stats - np.min(stats) + 0.1
        return np.log(stats)
    elif transform == "reciprocal":
        stats = stats - np.min(stats) + 1
        return 1 / stats
    elif transform == "sqrt":
        stats = stats - np.min(stats) + 0.1
        return np.sqrt(stats / np.std(stats[stats != None]))
    elif transform == "cbrt":
        stats = stats - np.min(stats) + 0.1
        return np.cbrt(stats)
    elif transform == "outlier":
        return stats
    else:
        raise NotImplementedError

def remove_outliears(stats, name):
    origin_size = stats.shape[0]

    if name == "rating_std":
        stats = stats[stats[name] != 0.0]

    else:
        stats = stats[np.abs(scipy.stats.zscore(stats[name])) < 3]

    num_dropped = origin_size - stats.shape[0]
    if num_dropped > 0:
        print(f"    {origin_size - stats.shape[0]} outliers out of {origin_size} users are dropped.")

    return stats

def plot_transform(config, show):
    name = config["name"]
    new_name = f"{config["name"]}_{config["transform"]}"

    file_name = f"{results_path}/transform/{name}/{new_name}.png"
    if not show and os.path.exists(file_name):
        return

    fig = plt.figure(figsize=(25, 12.5))
    gs = gridspec.GridSpec(2, 4)

    transformation = partial(apply_transform, transform=config["transform"])

    stats = config["stats"].dropna(subset=name)
    stats = remove_outliears(stats, name)
    new_stats = pd.DataFrame({new_name: transformation(stats[name].values)})
    stats = pd.concat([stats, new_stats], axis=1)

    config["opt_desc"] = None
    min_bin, max_bin = np.min(stats[[new_name]]), np.max(stats[new_name])
    new_bins = np.linspace(min_bin, max_bin, 100)
    config["bins"] = new_bins

    desc = config["desc"]
    desc = f"{config["transform"].replace("_", " ")} of {desc}"

    ax1 = fig.add_subplot(gs[0, 0])
    plot_bar(ax1, stats, new_name, desc)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_qq(ax2, stats, new_name, desc)

    ax3 = fig.add_subplot(gs[1, 0])
    plot_box(ax3, stats, new_name, desc,
            opt_desc=config.get("opt_desc", None))

    ax4 = fig.add_subplot(gs[1, 1])
    plot_violin(ax4, stats, new_name, desc,
            opt_desc=config.get("opt_desc", None))

    ax5 = fig.add_subplot(gs[:, 2:])
    plot_hist(ax5, stats, new_name, desc,
            bins=config["bins"], opt_desc=config.get("opt_desc", None))

    plt.tight_layout()

    if show:
        plt.show()
    else:
        os.makedirs(f"{results_path}/transform/{name}", exist_ok=True)
        plt.savefig(file_name)
        plt.close()

def plot_pair(config, show):
    file_name = f"{results_path}/pair/{config["name_x"]}_{config["name_y"]}.png"
    if not show and os.path.exists(file_name):
        return

    fig = plt.figure(figsize=(25, 12.5))
    gs = gridspec.GridSpec(2, 4)

    stats = config["stats"]
    stats["ratio"] = stats[config["name_y"]] / stats[config["name_x"]]

    desc_x, desc_y = config["desc_x"], config["desc_y"]
    name_x, name_y = config["name_x"], config["name_y"]
    ratio_desc = f"ratio of {desc_y}\nover {desc_x}"

    ax1 = fig.add_subplot(gs[0, 0])
    plot_bar(ax1, stats, "ratio", ratio_desc)

    ratio_range = config["ratio_range"]
    if ratio_range is not None:
        stats = stats[np.logical_and(stats["ratio"] >= ratio_range[0], stats["ratio"] <= ratio_range[1])]

    ax2 = fig.add_subplot(gs[0, 1])
    plot_hist(ax2, stats, "ratio", ratio_desc,
            bins=config["bins"], opt_desc=config.get("opt_ratio_desc", None))

    ax3 = fig.add_subplot(gs[1, 0])
    plot_box(ax3, stats, "ratio", ratio_desc,
            opt_desc=config.get("opt_ratio_desc", None))

    ax4 = fig.add_subplot(gs[1, 1])
    plot_violin(ax4, stats, "ratio", ratio_desc,
            opt_desc=config.get("opt_ratio_desc", None))

    ax5 = fig.add_subplot(gs[:, 2:])
    plot_scatter(ax5, config["filtered_stats"], name_x, name_y, desc_x, desc_y,
            opt_desc=config.get("opt_scatter_desc", None))

    plt.tight_layout()

    if show:
        plt.show()
    else:
        os.makedirs(f"{results_path}/pair", exist_ok=True)
        plt.savefig(file_name)
        plt.close()

def plot_bar(ax, stats, name, desc):
    sns.set_style("whitegrid")

    ax.set_title(f"Bar plot of {desc} by user groups")
    ax.set_xlabel("User groups")
    ax.set_ylabel(desc)

    sns.barplot(
        stats, x="group", y=name, ax=ax,
        hue="group", order=order, palette=palette, alpha=0.8)

def plot_hist(ax, stats, name, desc, bins, opt_desc):
    sns.set_style("whitegrid")

    if opt_desc is None:
        ax.set_title(f"Histogram of {desc} by user groups")
    else:
        ax.set_title(f"Histogram of {desc} by user groups\n({opt_desc})")

    ax.set_xlabel(desc)
    ax.set_ylabel("Frequenty")

    for group in reversed(order):
        sns.histplot(
            stats[stats["group"] == group], x=name, ax=ax,
            color=palette[group], alpha=0.5, label=group, bins=bins)

    ax.legend()

def plot_box(ax, stats, name, desc, opt_desc):
    sns.set_style("whitegrid")

    if opt_desc is None:
        ax.set_title(f"Box plot of {desc} by user groups")
    else:
        ax.set_title(f"Box plot of {desc} by user groups\n({opt_desc})")

    ax.set_xlabel("User groups")
    ax.set_ylabel(desc)

    sns.boxplot(
        stats, x="group", y=name, ax=ax,
        hue="group", order=order, palette=palette)

def plot_violin(ax, stats, name, desc, opt_desc):
    sns.set_style("whitegrid")

    if opt_desc is None:
        ax.set_title(f"Violin plot of {desc} by user groups")
    else:
        ax.set_title(f"Violin plot of {desc} by user groups\n({opt_desc})")

    ax.set_xlabel("User groups")
    ax.set_ylabel(desc)

    sns.violinplot(
        stats, x="group", y=name, ax=ax,
        hue="group", order=order, palette=palette, alpha=0.8)

def plot_qq(ax, stats, name, desc):
    sns.set_style("whitegrid")

    ax.set_title(f"Violin plot of {desc} by user groups")
    ax.set_xlabel(f"Quantiles of {desc}")
    ax.set_ylabel("Quantiles of standard normal")

    for group in order:
        group_stats = stats[stats["group"] == group][name].dropna().values
        group_stats = (group_stats - np.mean(group_stats)) / np.std(group_stats)

        sm.qqplot(group_stats, line="s", ax=ax, label=group,
                  markerfacecolor=palette[group], markeredgecolor=palette[group],
                  markersize=3, alpha=0.8)

    for i in range(6):
        ax.get_lines()[i].set_color("black")
    ax.legend()

def plot_scatter(ax, stats, name_x, name_y, desc_x, desc_y, opt_desc):
    sns.set_style("whitegrid")

    if opt_desc is None:
        ax.set_title(f"Scatter plot of {desc_x} and {desc_y} by user groups")
    else:
        ax.set_title(f"Scatter plot of {desc_x} and {desc_y} by user groups\n({opt_desc})")

    ax.set_xlabel(desc_x)
    ax.set_ylabel(desc_y)

    sns.scatterplot(stats, x=name_x, y=name_y, hue="group", style="group", palette=palette)