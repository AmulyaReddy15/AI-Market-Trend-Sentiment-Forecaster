import pandas as pd
import re

# -----------------------------
# LOAD FILES
# -----------------------------
rapid_df = pd.read_csv(r"final data/rapid_api_reviews_final.csv")
reviews_df = pd.read_csv(r"final data/category_wise_lda_output_with_topic_labels.csv")

# -----------------------------
# PROCESS RAPID FILE
# -----------------------------

# Combine title + text
rapid_df["review_text"] = (
    rapid_df["review_title"].fillna("") + " " +
    rapid_df["review_text"].fillna("")
).str.strip()

# Rename label -> category
rapid_df.rename(columns={"label": "category"}, inplace=True)

# Extract date from text
rapid_df["review_date"] = rapid_df["review_date"].astype(str).apply(
    lambda x: re.search(r"on (.*)", x).group(1) if re.search(r"on (.*)", x) else None
)

# Format date as DD-MM-YYYY
rapid_df["review_date"] = pd.to_datetime(
    rapid_df["review_date"], errors="coerce"
).dt.strftime("%d-%m-%Y")

# Generate sentiment from rating
def rating_to_sentiment(r):
    if r >= 4:
        return "Positive"
    elif r == 3:
        return "Neutral"
    else:
        return "Negative"

rapid_df["sentiment_label"] = rapid_df["rating"].apply(rating_to_sentiment)

# Keep final columns
rapid_final = rapid_df[
    ["source", "review_text", "rating", "review_date", "sentiment_label", "category"]
]

# -----------------------------
# PROCESS REVIEWS (LDA) FILE
# -----------------------------

# Standardize column names if needed
reviews_df.rename(
    columns={
        "sentiment": "sentiment_label"
    },
    inplace=True
)

# Format date
reviews_df["review_date"] = pd.to_datetime(
    reviews_df["review_date"], errors="coerce"
).dt.strftime("%d-%m-%Y")

reviews_final = reviews_df[
    ["source", "review_text", "rating", "review_date", "sentiment_label", "category"]
]

# -----------------------------
# CONCATENATE FILES
# -----------------------------
final_df = pd.concat(
    [rapid_final, reviews_final],
    ignore_index=True
)

# -----------------------------
# SAVE OUTPUT
# -----------------------------
final_df.to_csv("review_rapid_combined.csv", index=False)

print("✅ review_rapid_combined.csv saved successfully")
