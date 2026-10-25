import datetime

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import pandas as pd
from pyzipcode import ZipCodeDatabase

interim_path = "../../data/01-interim"
processed_path = "../../data/02-processed"

analyzer = SentimentIntensityAnalyzer()

def get_gmap_id_to_city():
    zcdb = ZipCodeDatabase()

    restaurants = pd.read_csv(f"{interim_path}/restaurants.csv")
    gmap2city = {}
    for gmap_id, address in restaurants[["gmap_id", "address"]].values:
        zipcode = address.split()[-1]
        try:
            city = zcdb[int(zipcode)].city
            gmap2city[gmap_id] = city
        except:
            gmap2city[gmap_id] = None

    return gmap2city

def convert_to_year_month(unix_time):
    dt = datetime.datetime.fromtimestamp(unix_time / 1000, datetime.timezone.utc)
    year = dt.year
    month = dt.month

    return f"{year}_{month}"

def convert_to_num(unix_time):
    dt = datetime.datetime.fromtimestamp(unix_time / 1000, datetime.timezone.utc)
    year = dt.year
    month = dt.month

    return f"{year}_{month}"

def get_text_length(text):
    return len(text.strip().split())

def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return scores["compound"]

def get_diversity(text):
    tokens = text.lower().split()
    return len(tokens) / len(set(tokens))