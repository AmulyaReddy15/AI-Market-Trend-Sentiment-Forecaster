import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import re

# -----------------------------
# FILE PATH
# -----------------------------
CSV_PATH = r"final data\rapid_api_reviews_final.csv"

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)

# -----------------------------
# COMBINE REVIEW TEXT
# (change column names if needed)
# -----------------------------
df["combined_review"] = (
    df["review_title"].fillna("") + " " +
    df["review_text"].fillna("")
)

# -----------------------------
# LOAD FINBERT MODEL
# -----------------------------
MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

labels = ["Negative", "Neutral", "Positive"]

# -----------------------------
# SENTIMENT PREDICTION
# -----------------------------
sentiments = []

for text in tqdm(df["combined_review"], desc="Analyzing Sentiment"):
    if not text.strip():
        sentiments.append("Neutral")
        continue

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    sentiments.append(labels[pred])

df["sentiment_label"] = sentiments

# -----------------------------
# (DD-MM-YYYY)
# -----------------------------
df["review_date"] = df["review_date"].astype(str).apply(
    lambda x: re.search(r"on\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", x).group(1)
    if re.search(r"on\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", x)
    else None
)

# -----------------------------
# CONVERT TO DD-MM-YYYY
# -----------------------------
df["review_date"] = pd.to_datetime(
    df["review_date"],
    format="%B %d, %Y",
    errors="coerce"
).dt.strftime("%d-%m-%Y")

# -----------------------------
# SAVE BACK
# -----------------------------
df.to_csv("rapid_file_new.csv", index=False)

print(" Sentiment labels generated and review_date cleaned successfully")
