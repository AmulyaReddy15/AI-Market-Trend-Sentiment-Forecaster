import requests
import pandas as pd
from datetime import datetime
import time
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import notification.notification as notification

# ============================================================
# MODEL CONFIG

MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ============================================================
# TEXT CLEANING FUNCTION

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.encode("utf-8", "ignore").decode("utf-8")
    return text.strip()

# ============================================================
# SENTIMENT FUNCTION

def get_sentiment(text):
    if pd.isna(text) or text.strip() == "":
        return "Neutral"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        sentiment_idx = torch.argmax(probs).item()

    label_map = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }
    return label_map[sentiment_idx]

# ============================================================
# REDDIT FETCH FUNCTION

def fetch_reddit_data(labels):
    url = "https://www.reddit.com/search.json"
    headers = {
        "User-Agent": "ConsumerTrendAnalysisBot/1.0"
    }

    rows = []

    for label in tqdm(labels, desc="Fetching Reddit Data"):
        params = {
            "q": label.replace("_", " "),
            "limit": 100
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f" Failed for {label}")
            continue

        data = response.json()

        for post in data["data"]["children"]:
            p = post["data"]

            rows.append({
                "post_id": p.get("id"),   # 🔑 UNIQUE KEY
                "source": "Reddit",
                "category_label": label,
                "search_query": label.replace("_", " "),
                "title": p.get("title", ""),
                "selftext": p.get("selftext", ""),
                "subreddit": p.get("subreddit", ""),
                "score": p.get("score", 0),
                "num_comments": p.get("num_comments", 0),
                "created_date": datetime.utcfromtimestamp(
                    p.get("created_utc", 0)
                ),
                "collected_at": datetime.utcnow()
            })

        time.sleep(2)  # polite delay

    return pd.DataFrame(rows)

# ============================================================
# MAIN PIPELINE

def get_reddit_data():
    error=[]
    OUTPUT_FILE = "final_data/reddit_data_with_sentiment.xlsx"

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
    # FETCH DATA

    new_df = fetch_reddit_data(labels)

    # Remove empty posts
    new_df = new_df[new_df["selftext"].str.strip() != ""].reset_index(drop=True)

    # Clean text fields
    for col in ["title", "selftext", "subreddit"]:
        new_df[col] = new_df[col].apply(clean_text)

    # -----------------------------
    # SENTIMENT ANALYSIS

    new_df["combined_text"] = (
        new_df["title"].fillna("") + ". " +
        new_df["selftext"].fillna("")
    )

    tqdm.pandas(desc="Applying Sentiment")
    new_df["sentiment_label"] = new_df["combined_text"].progress_apply(get_sentiment)

    new_df.drop(columns=["combined_text"], inplace=True)

# -----------------------------
# SAVE CURRENT WEEK DATA (SEPARATE FILE)

    # os.makedirs(os.path.dirname(WEEKLY_FILE), exist_ok=True)
    new_df.to_excel("reddit_current_week_data.xlsx", index=False)
    print(" Weekly Reddit data saved ")

    # -----------------------------
    # MERGE + OVERRIDE DUPLICATES

    if os.path.exists(OUTPUT_FILE):
        old_df = pd.read_excel(OUTPUT_FILE)

        final_df = pd.concat([old_df, new_df], ignore_index=True)

        # 🔑 Override duplicate posts
        final_df.drop_duplicates(
            subset="post_id",
            keep="last",
            inplace=True
        )
    else:
        final_df = new_df

    # -----------------------------
    # SAVE OVERWRITE 
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    final_df.to_excel(OUTPUT_FILE, index=False)

    print(f"✅ Reddit pipeline completed. Total records: {len(final_df)}")

    if error:
        notification.send_mail(
        "Reddit Data Failed",
        "\n".join(error[:5])  # only first 5 errors
         )
    else:
        notification.send_mail(
        "Reddit Data Extracted Successfully",
        "Pipeline completed successfully"
    )
