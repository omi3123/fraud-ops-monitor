from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, ASSET_DIR, CASE_QUEUE_PATH, DATA_DIR, DEMO_DATA_PATH, ENRICHED_DATA_PATH, RAW_DATA_PATH, SEED
from src.data_prep import build_demo_dataset, enrich_transactions, load_raw_data
from src.modeling import fit_hybrid_model



def ensure_dirs() -> None:
    for path in [ARTIFACT_DIR, ASSET_DIR, DATA_DIR]:
        path.mkdir(parents=True, exist_ok=True)



def build_assets() -> None:
    ensure_dirs()
    raw = load_raw_data(RAW_DATA_PATH)
    enriched = enrich_transactions(raw, seed=SEED)
    fraud_all = enriched[enriched["Class"] == 1]
    legit_sample = enriched[enriched["Class"] == 0].sample(n=80_000, random_state=SEED)
    train_ready = (
        pd.concat([fraud_all, legit_sample], axis=0)
        .sort_values("event_ts")
        .reset_index(drop=True)
    )
    split_idx = int(len(train_ready) * 0.8)
    train_df = train_ready.iloc[:split_idx].copy()
    test_df = train_ready.iloc[split_idx:].copy()

    model = fit_hybrid_model(train_df, test_df)
    model.save(ARTIFACT_DIR)

    demo_df = build_demo_dataset(enriched, legit_rows=50_000)
    scored_demo = model.score(demo_df)
    scored_demo.to_csv(DEMO_DATA_PATH, index=False)

    alert_queue = (
        scored_demo.loc[:, [
            "transaction_id",
            "event_ts",
            "customer_id",
            "Amount",
            "channel",
            "merchant_category",
            "country",
            "priority_band",
            "alert_priority_score",
            "fraud_probability",
            "anomaly_score",
            "rule_hits",
            "risk_summary",
            "Class",
        ]]
        .sort_values(["alert_priority_score", "Amount"], ascending=[False, False])
        .head(500)
        .reset_index(drop=True)
    )
    alert_queue.to_csv(CASE_QUEUE_PATH, index=False)

    template_cols = [
        "Time", "Amount", *[f"V{i}" for i in range(1, 29)], "account_age_days", "amount_vs_profile",
        "minutes_since_prev_tx", "amount_delta_abs", "customer_tx_index", "customer_recent_fraud_count",
        "prior_chargeback_count", "merchant_risk_score", "is_cross_border", "rapid_repeat_flag",
        "new_account_flag", "high_amount_flag", "night_tx_flag", "hour", "channel", "merchant_category",
        "device_risk_tier", "risk_segment", "country", "home_country", "day_name"
    ]
    sample_template = scored_demo[template_cols].head(25)
    sample_template.to_csv(DATA_DIR / "demo_input_template.csv", index=False)

    build_charts(scored_demo, model.metrics)



def build_charts(scored_demo: pd.DataFrame, metrics: dict) -> None:
    recent = scored_demo.sort_values("event_ts").copy()
    recent["event_ts"] = pd.to_datetime(recent["event_ts"])

    hourly = (
        recent.assign(hour_bucket=recent["event_ts"].dt.floor("H"))
        .groupby("hour_bucket", as_index=False)
        .agg(
            alert_volume=("transaction_id", "count"),
            avg_priority=("alert_priority_score", "mean"),
            confirmed_fraud=("Class", "sum"),
        )
        .tail(48)
    )

    plt.figure(figsize=(12, 5))
    plt.plot(hourly["hour_bucket"], hourly["alert_volume"])
    plt.title("Alert Volume Over Time")
    plt.xlabel("Hour")
    plt.ylabel("Transactions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "alert_volume_over_time.png", dpi=180)
    plt.close()

    band_counts = (
        recent["priority_band"].value_counts().reindex(["critical", "high", "medium", "low"]).fillna(0)
    )
    plt.figure(figsize=(8, 5))
    plt.bar(band_counts.index, band_counts.values)
    plt.title("Priority Band Distribution")
    plt.xlabel("Priority band")
    plt.ylabel("Transactions")
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "priority_band_distribution.png", dpi=180)
    plt.close()

    score_bins = pd.cut(recent["alert_priority_score"], bins=[0, 35, 60, 80, 100], include_lowest=True)
    fraud_rate = recent.groupby(score_bins, observed=False)["Class"].mean().fillna(0)
    plt.figure(figsize=(8, 5))
    plt.plot([str(x) for x in fraud_rate.index], fraud_rate.values, marker="o")
    plt.title("Observed Fraud Rate by Alert Score Band")
    plt.xlabel("Alert score band")
    plt.ylabel("Observed fraud rate")
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "fraud_rate_by_alert_band.png", dpi=180)
    plt.close()

    hero = {
        "transactions": int(len(recent)),
        "fraud_rate": float(recent["Class"].mean()),
        "avg_priority_score": float(recent["alert_priority_score"].mean()),
        "critical_alerts": int((recent["priority_band"] == "critical").sum()),
        "review_queue": int((recent["needs_review"] == 1).sum()),
        **metrics,
    }
    (ASSET_DIR / "hero_metrics.json").write_text(json.dumps(hero, indent=2), encoding="utf-8")


if __name__ == "__main__":
    build_assets()
    print("Build complete.")
