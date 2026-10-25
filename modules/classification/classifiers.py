import sys
import os

sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from modules.utils.load_dataset import *

palette = {"critical": "#a02121", "neutral": "#54d582", "generous": "#5497d5"}
order = ["critical", "neutral", "generous"]

def classify_users(stats, models, show=False):
    vars = list(stats.columns)[2:]
    values = stats[vars].values
    labels = stats["group"].values

    x_train, x_test, y_train, y_test = train_test_split(values, labels, test_size=0.1, random_state=0)

    sns.set_style("whitegrid")
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    plt.suptitle("Confusion matrices for classifiers on test dataset.")

    accuracies = []
    for i in range(len(models)):
        name, model = models[i]
        print(f"Classifying users with {name}")

        model = model.fit(x_train, y_train)

        ax = axes.flatten()[i]
        ax.grid(False)
        ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax, cmap="GnBu")
        ax.set_title(name)

        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        train_accuracy = np.sum(y_train_pred == y_train) / y_train.shape[0]
        test_accuracy = np.sum(y_test_pred == y_test) / y_test.shape[0]

        accuracies.append((name, train_accuracy, "train"))
        accuracies.append((name, test_accuracy, "test"))

    plt.tight_layout()

    if show:
        plt.show()
    else:
        os.makedirs(f"{results_path}/classification", exist_ok=True)
        plt.savefig(f"{results_path}/classification/confusion_matrix.png")
        plt.close()

    df = pd.DataFrame(accuracies, columns=["model", "accuracy", "split"])
    print(df)

    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))
    plt.title("Comparison of train and test accuracies with different models.")
    sns.barplot(df, x="model", y="accuracy", hue="split", alpha=0.8)

    plt.xlabel("Models")
    plt.xlabel("Accuracy")

    plt.ylim(0.5, 0.8)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(f"{results_path}/classification/accuracies.png")
        plt.close()

if __name__ == "__main__":
    stats = load_stats()

    models = [
        ("Logisitc Regression", LogisticRegression(max_iter=10000)),
        ("Support Vector Machine", SVC(kernel="rbf")),
        ("Random Forest", RandomForestClassifier(max_depth=5)),
        ("Ada Boost", AdaBoostClassifier())
    ]

    classify_users(stats, models)
