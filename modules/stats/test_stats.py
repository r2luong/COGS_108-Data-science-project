import sys
import json

sys.path.append("../../")

import numpy as np
import scipy

from modules.utils.load_dataset import *
from modules.utils.stats_utils import *
from modules.utils.plot_utils import *

results_path = "../../results"

np.random.seed(0)

class StatisticsTester():
    def __init__(self, stats):
        self.stats = stats
        self.groups = np.unique(self.stats["group"])

        self.transforms = ["outlier", "log", "reciprocal", "sqrt", "cbrt"]

        self.p_thresh = 0.01

    def goodness_of_fit(self, values, name):
        p_values = []
        for group in self.groups:
            group_stats = values[values["group"] == group][name]
            params = {"loc": np.mean(group_stats), "scale": np.std(group_stats)}

            res = scipy.stats.goodness_of_fit(scipy.stats.norm, group_stats, known_params=params)
            p_values.append(res.pvalue)

        p_value = min(p_values)

        return p_value

    def two_sample_t_test(self, group_0, group_1):
        res = scipy.stats.ttest_ind(group_0, group_1, equal_var=False)
        return res.pvalue, res.statistic

    def mann_whitney_u_test(self, group_0, group_1):
        res = scipy.stats.mannwhitneyu(group_0, group_1)
        return res.pvalue, res.statistic

    def test_raw_normal(self):
        with open(f"{results_path}/stats.txt", "r") as f:
            stats_list = f.read().split()

        results = {}
        for stats in stats_list:
            print(f"Testing raw normal for {stats}")
            values = self.stats[["group", stats]]
            p_value = self.goodness_of_fit(values, stats)
            results[stats] = {"raw_normal": int(p_value >= self.p_thresh), "raw_normal_p": p_value}

        with open(f"{results_path}/distribution.json", "w") as f:
            json.dump(results, f)

    def test_transform_normal(self):
        with open(f"{results_path}/distribution.json", "r") as f:
            results = json.load(f)

        old_results = results
        for key, values in old_results.items():
            if results[key]["raw_normal"]:
                continue

            if "transform_normal" in results[key].keys():
                continue

            for transform in self.transforms:
                print(f"Testing raw normal for {key} with {transform}")

                values = self.stats[["group", key]].dropna()
                values.iloc[:, 1] = apply_transform(values[key].values, transform=transform)

                p_value = self.goodness_of_fit(values, key)
                if p_value >= self.p_thresh:
                    results[key]["transform_normal"] = 1
                    results[key]["transform_normal_p"] = float(p_value)
                    results[key]["transform"] = transform

            if "transform_normal" not in results[key].keys():
                results[key]["transform_normal"] = 0

            with open(f"{results_path}/distribution.json", "w") as f:
                json.dump(results, f)

    def independence_tests(self, dists):
        pairs = [(self.groups[i], self.groups[j]) for i in range(3) for j in range(i+1, 3)]
        results = []

        for key, dist in dists.items():
            values = self.stats[["group", key]].dropna()

            if dist["normal"] == 1:
                t_test = True
            elif dist["transformed_normal"] == 1:
                t_test = True
                values[key] = apply_transform(values[key].values, transform=dist["test_transform"])
            else:
                t_test = False

            for (group_0, group_1) in pairs:
                values_0 = values[values["group"] == group_0][key].dropna().values
                values_1 = values[values["group"] == group_1][key].dropna().values

                if t_test:
                    t_test_p, t_test_stat = self.two_sample_t_test(values_0, values_1)
                    log_t_test_p = -np.where(t_test_p == 0, -500, np.log10(t_test_p))
                else:
                    t_test_p = None
                    log_t_test_p = None
                    t_test_stat = None

                u_test_p, u_test_stat = self.mann_whitney_u_test(values_0, values_1)
                log_u_test_p = -np.where(u_test_p == 0, -500, np.log10(u_test_p))

                results.append((key, f"{group_0}_{group_1}",
                                t_test_p, u_test_p,
                                t_test_stat, u_test_stat,
                                log_t_test_p, log_u_test_p
                ))

        results = pd.DataFrame(results, columns=[
            "name", "pair", "t_test_p", "u_test_p", "t_test_stat", "u_test_stat", "log_t_test_p", "log_u_test_p"
        ])
        print(results)

        os.makedirs(f"{results_path}/tests", exist_ok=True)
        results.to_csv(f"{results_path}/tests/results.csv", index=False)

        return results

    def plot_test_results(self, results, show):
        _, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))

        sns.set_style("whitegrid")
        plt.suptitle(f"Comparisons of p-values of test results (red dot line corresponds to 0.01)")

        axes[0].set_title("Comparison of log of p values for two sample t test.\n(if negative log of p value is 0, then it is replaced as 500)")
        axes[0].set_xlabel("Negative log of p value")
        axes[0].set_ylabel("Statistic")
        axes[0].axvline(-np.log(self.p_thresh), color="red", linestyle="dashed")
        sns.barplot(results, y="name", x="log_t_test_p", hue="pair", ax=axes[0], alpha=0.8)

        axes[1].set_title("Comparison of p values for Mann Whitney U test.")
        axes[1].set_xlabel("Negative log of p value")
        axes[1].set_ylabel("Statistic")
        axes[1].axvline(-np.log(self.p_thresh), color="red", linestyle="dashed")
        sns.barplot(results, y="name", x="log_u_test_p", hue="pair", ax=axes[1], alpha=0.8)

        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(f"{results_path}/tests/p_comparison.png")
            plt.close()

        _, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
        plt.suptitle(f"Comparisons of p-values of test results (red dot line corresponds to critical value for 0.01)")

        axes[0].set_title("Comparison of test statistics for two sample t test.")
        axes[0].set_xlabel("Value of test statistic")
        axes[0].set_ylabel("Statistic")
        axes[0].axvline(-2.57644, color="red", linestyle="dashed")
        axes[0].axvline(2.57644, color="red", linestyle="dashed")
        sns.barplot(results, y="name", x="t_test_stat", hue="pair", ax=axes[0], alpha=0.8)

        axes[1].set_title("Comparison of test statistics for Mann Whitney U test.")
        axes[1].set_xlabel("Value of test statistic")
        axes[1].set_ylabel("Statistic")
        sns.barplot(results, y="name", x="u_test_stat", hue="pair", ax=axes[1], alpha=0.8)

        plt.tight_layout()

        if show:
            plt.show()
        else:
            plt.savefig(f"{results_path}/tests/stat_comparison.png")
            plt.close()

    def test_statistics(self, dists, show=False):
        if not os.path.exists(f"{results_path}/distribution.json"):
            self.test_raw_normal()
            self.test_transform_normal()

        results = self.independence_tests(dists)
        self.plot_test_results(results, show)

if __name__ == "__main__":
    with open(f"{results_path}/manual_distribution.json", "r") as f:
        dists = json.load(f)

    stats = load_stats()
    tester = StatisticsTester(stats)
    tester.test_statistics(dists)