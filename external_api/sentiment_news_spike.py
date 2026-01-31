import pandas as pd
from notification import notification
# -----------------------------
# CONFIG
# -----------------------------


def new_sentiment_spike(df):
    DATE_COL = "published_at"
    SENTIMENT_COL = "sentiment_label"
    CATEGORY_COL = "category"

    WEEK_WINDOW = 2  # rolling over 2 weeks
    SPIKE_THRESHOLD = 0.20       # only major spikes
    TREND_SHIFT_THRESHOLD = 0.20 # only major trend reversals

    LABEL_MAP = {
        "Positive": 1,
        "Neutral": 0,
        "Negative": -1
    }

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)

    # Map sentiment to numeric
    df["sentiment_score"] = df[SENTIMENT_COL].map(LABEL_MAP)

    # -----------------------------
    # WEEKLY AGGREGATION
    # -----------------------------
    weekly_cat = (
        df.groupby([CATEGORY_COL, df[DATE_COL].dt.to_period("W")])["sentiment_score"]
        .mean()
        .reset_index(name="weekly_sentiment")
    )
    weekly_cat[DATE_COL] = weekly_cat[DATE_COL].dt.start_time


    # -----------------------------
    # ROLLING WINDOWS
    # -----------------------------
    weekly_cat["rolling_avg"] = weekly_cat.groupby(CATEGORY_COL)["weekly_sentiment"] \
                                        .rolling(WEEK_WINDOW, min_periods=1).mean() \
                                        .reset_index(level=0, drop=True)

    weekly_cat["prev_rolling_avg"] = weekly_cat.groupby(CATEGORY_COL)["rolling_avg"].shift(1)
    weekly_cat["delta"] = weekly_cat["rolling_avg"] - weekly_cat["prev_rolling_avg"]
    weekly_cat["prev_delta"] = weekly_cat.groupby(CATEGORY_COL)["delta"].shift(1)

    # -----------------------------
    # DETECT ALERTS
    # -----------------------------
    alerts = []

    for _, row in weekly_cat.iterrows():
        if pd.isna(row["delta"]):
            continue

        #  Negative spike
        if row["delta"] <= -SPIKE_THRESHOLD:
            alerts.append({
                "date": row[DATE_COL],
                "category": row[CATEGORY_COL],
                "type": "NEGATIVE SPIKE",
                "change": round(row["delta"], 3)
            })

        #  Positive spike
        if row["delta"] >= SPIKE_THRESHOLD:
            alerts.append({
                "date": row[DATE_COL],
                "category": row[CATEGORY_COL],
                "type": "POSITIVE SPIKE",
                "change": round(row["delta"], 3)
            })

        #  Trend shift (direction change)
        if pd.notna(row["prev_delta"]) and (row["delta"] * row["prev_delta"] < 0) and \
        (abs(row["delta"]) >= TREND_SHIFT_THRESHOLD):
            alerts.append({
                "date": row[DATE_COL],
                "category": row[CATEGORY_COL],
                "type": "TREND SHIFT",
                "change": round(row["delta"], 3)
            })

    # -----------------------------
    # OUTPUT ALERTS
    # -----------------------------
    alert_df = pd.DataFrame(alerts)

    if alert_df.empty:
        print(" No major weekly sentiment spikes or trend shifts detected.")
    else:
        print(" MAJOR WEEKLY SENTIMENT ALERTS")
        print(alert_df.sort_values(["date", "category"]).reset_index(drop=True))
        
    return alert_df

if __name__ == "__main__":

    # -----------------------------
    # LOAD CSV
    # -----------------------------
    input_csv = "final data/news_data_with_sentiment.csv"
    df = pd.read_csv(input_csv)

    # -----------------------------
    # RUN SENTIMENT SPIKE DETECTION
    # -----------------------------
    alert_df = new_sentiment_spike(df)

    # -----------------------------
    # SAVE TO EXCEL
    # -----------------------------
    output_excel = "weekly_sentiment_alerts.xlsx"

    if not alert_df.empty:
        alert_df.to_excel(output_excel, index=False)
        print(f" Alerts saved to {output_excel}")
    else:
        print(" No alerts to save.")

    # -----------------------------
    # SEND EMAIL
    # -----------------------------
    if not alert_df.empty:
        notification.send_mail(
            "Weekly Sentiment Spike Alerts",
            "Major sentiment spikes and trend shifts detected. See attached report.",
            alert_df
        )

