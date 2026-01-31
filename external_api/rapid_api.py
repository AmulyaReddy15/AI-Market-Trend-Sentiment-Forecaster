import requests
import pandas as pd
import os
from tqdm import tqdm
import time

from rapid import rapid_sentiment_spike
import notification.notification as notification

# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_FILE = "rapid_file_new.csv"

RAPIDAPI_KEY = "5d130f4acbmsh22294a76c92ee8ap1b6694jsn425fd0ef979c"

HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"
}

SEARCH_URL = "https://real-time-amazon-data.p.rapidapi.com/search"
REVIEW_URL = "https://real-time-amazon-data.p.rapidapi.com/product-reviews"

COUNTRY = "US"
SEARCH_PAGE = 1
REVIEW_PAGE = 1

# -----------------------------
# CATEGORIES
# -----------------------------
labels = [
    "Electricals_Power_Backup",
    "Home_Appliances",
    "Kitchen_Appliances",
    "Furniture",
    "Home_Storage_Organization",
    "Computers_Tablets",
    "Mobile_Accessories",
    "Wearables",
    "TV_Audio_Entertainment",
    "Networking_Devices",
    "Toys_Kids",
    "Gardening_Outdoor",
    "Kitchen_Dining",
    "Mens_Clothing",
    "Footwear",
    "Beauty_Personal_Care",
    "Security_Surveillance",
    "Office_Printer_Supplies",
    "Software",
    "Fashion_Accessories"
]

# -----------------------------
# API HELPERS
# -----------------------------
def search_products(query):
    params = {
        "query": query,
        "page": SEARCH_PAGE,
        "country": COUNTRY,
        "sort_by": "RELEVANCE"
    }

    r = requests.get(SEARCH_URL, headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json().get("data", {}).get("products", [])

def fetch_reviews(asin):
    params = {
        "asin": asin,
        "country": COUNTRY,
        "page": REVIEW_PAGE,
        "sort_by": "TOP_REVIEWS"
    }

    r = requests.get(REVIEW_URL, headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json().get("data", {}).get("reviews", [])

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def rapid_get_data():
    try:
        all_reviews = []

        for label in tqdm(labels, desc="Categories"):
            try:
                products = search_products(label)
            except Exception as api_err:
                print(f"Skipping '{label}' due to API error: {api_err}")
                continue

            for product in products[:2]:
                asin = product.get("asin")
                if not asin:
                    continue

                try:
                    reviews = fetch_reviews(asin)
                    time.sleep(1.5)
                except Exception as review_err:
                    print(f"Skipping ASIN {asin}: {review_err}")
                    continue

                for r in reviews:
                    all_reviews.append({
                        "source": "Amazon",
                        "label": label,
                        "search_keyword": label,
                        "review_id": r.get("review_id"),
                        "review_title": r.get("review_title"),
                        "review_text": r.get("review_text"),
                        "rating": r.get("rating"),
                        "author": r.get("reviewer_name"),
                        "verified_purchase": r.get("verified_purchase"),
                        "review_date": r.get("review_date"),
                        "sentiment_label": r.get("sentiment_label")  # already exists
                    })

        df = pd.DataFrame(all_reviews)

        # -------------------------------------------------
        # DECIDE INPUT FOR SPIKE ANALYSIS
        # -------------------------------------------------
        if df.empty:
            print("No new Rapid data fetched. Using existing file.")

            # if not os.path.exists(OUTPUT_FILE):
            #     notification.send_mail(
            #         "Rapid Sentiment Update",
            #         "No Rapid data available yet to analyze."
            #     )
            #     return

            analysis_df = pd.read_csv(OUTPUT_FILE)

        else:
            print("New Rapid data fetched. Appending to existing file.")

            if os.path.exists(OUTPUT_FILE):
                df.to_csv(OUTPUT_FILE, mode="a", index=False, header=False)
            else:
                df.to_csv(OUTPUT_FILE, index=False)

            analysis_df = pd.read_csv(OUTPUT_FILE)

        # -----------------------------
        # SENTIMENT SPIKE DETECTION
        # -----------------------------
        alert_df = rapid_sentiment_spike.sentiment_spike(analysis_df)

        if not alert_df.empty:
            notification.send_mail(
                "Rapid Sentiment Spike Alert",
                "Please find the attached sentiment spike report.",
                alert_df
            )
        else:
            notification.send_mail(
                "Rapid Sentiment Update",
                "No major weekly rapid sentiment spikes or trend shifts detected."
            )

    except Exception as e:
        notification.send_mail(
            "Rapid Data Alert",
            f"Rapid pipeline failed. Reason: {e}"
        )

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    rapid_get_data()
