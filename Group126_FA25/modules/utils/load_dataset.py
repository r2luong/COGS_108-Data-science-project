import pandas as pd

processed_path = "../../data/02-processed"
results_path = "../../results"

def load_user_reviews():
    critical = pd.read_csv(f"{processed_path}/critical_user_reviews.csv")
    generous = pd.read_csv(f"{processed_path}/generous_user_reviews.csv")
    neutral = pd.read_csv(f"{processed_path}/neutral_user_reviews.csv")

    return critical, generous, neutral

def load_combined_reviews():
    critical, generous, neutral = load_user_reviews()

    critical["group"] = critical["user_id"].apply(lambda x: "critical")
    generous["group"] = generous["user_id"].apply(lambda x: "generous")
    neutral["group"] = neutral["user_id"].apply(lambda x: "neutral")

    combined = pd.concat([critical, neutral, generous], axis=0)

    return combined

def load_stats():
    stats = pd.read_csv(f"{results_path}/stats.csv")
    return stats