from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import BASELINE_START, RAW_DATA_PATH, RAW_DATA_URL, SEED


COUNTRIES = np.array([
    "France",
    "Germany",
    "Spain",
    "Italy",
    "Netherlands",
    "Belgium",
    "Portugal",
    "Poland",
    "Romania",
    "Turkey",
    "UAE",
    "Singapore",
])


def download_raw_data(destination: Path | str = RAW_DATA_PATH, url: str = RAW_DATA_URL) -> Path:
    """Download the public credit card fraud dataset.

    The app ships with demo data, but this helper lets users pull the full raw file.
    """
    import requests

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination



def load_raw_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run scripts/download_real_data.py first."
        )
    df = pd.read_csv(path)
    expected = {"Time", "Amount", "Class"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df.sort_values("Time").reset_index(drop=True)



def _assign_customers(df: pd.DataFrame, rng: np.random.Generator, n_customers: int = 12000) -> pd.DataFrame:
    customer_ids = np.array([f"CUST-{i:05d}" for i in range(1, n_customers + 1)])
    risk_segment = rng.choice(["low", "medium", "high"], size=n_customers, p=[0.68, 0.24, 0.08])
    home_country = rng.choice(
        COUNTRIES[:8],
        size=n_customers,
        p=[0.18, 0.18, 0.14, 0.14, 0.12, 0.08, 0.08, 0.08],
    )
    avg_ticket = np.clip(rng.lognormal(mean=4.1, sigma=0.75, size=n_customers), 8, 800)
    base_account_age = rng.integers(20, 3650, size=n_customers)

    profiles = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "risk_segment": risk_segment,
            "home_country": home_country,
            "profile_avg_ticket": avg_ticket,
            "base_account_age": base_account_age,
        }
    )

    risk_pool = profiles.index[profiles["risk_segment"].isin(["medium", "high"])].to_numpy()
    normal_pool = profiles.index[profiles["risk_segment"] == "low"].to_numpy()

    fraud_mask = df["Class"].to_numpy() == 1
    cust_idx = np.empty(len(df), dtype=int)
    legit_idx = np.where(~fraud_mask)[0]
    fraud_idx = np.where(fraud_mask)[0]

    cust_idx[legit_idx] = rng.integers(0, n_customers, size=len(legit_idx))
    fraud_from_risk = rng.random(len(fraud_idx)) < 0.72
    cust_idx[fraud_idx[fraud_from_risk]] = rng.choice(risk_pool, size=fraud_from_risk.sum(), replace=True)
    cust_idx[fraud_idx[~fraud_from_risk]] = rng.choice(normal_pool, size=(~fraud_from_risk).sum(), replace=True)

    enriched = df.copy()
    enriched["customer_id"] = customer_ids[cust_idx]
    enriched["risk_segment"] = risk_segment[cust_idx]
    enriched["home_country"] = home_country[cust_idx]
    enriched["profile_avg_ticket"] = avg_ticket[cust_idx]
    enriched["account_age_days"] = base_account_age[cust_idx] + (enriched["Time"] // 86400).astype(int)
    return enriched



def _assign_operational_context(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    fraud_mask = df["Class"].to_numpy() == 1
    size = len(df)

    def draw(labels, legit_p, fraud_p):
        arr = np.empty(size, dtype=object)
        arr[~fraud_mask] = rng.choice(labels, size=(~fraud_mask).sum(), p=legit_p)
        arr[fraud_mask] = rng.choice(labels, size=fraud_mask.sum(), p=fraud_p)
        return arr

    df = df.copy()
    df["channel"] = draw(
        np.array(["card_present", "e_commerce", "mobile_app", "atm"]),
        [0.47, 0.25, 0.20, 0.08],
        [0.10, 0.58, 0.24, 0.08],
    )
    df["merchant_category"] = draw(
        np.array(["grocery", "retail", "travel", "electronics", "digital_goods", "bill_pay", "cash_withdrawal"]),
        [0.24, 0.22, 0.08, 0.11, 0.06, 0.21, 0.08],
        [0.06, 0.14, 0.14, 0.22, 0.29, 0.05, 0.10],
    )

    device_risk = np.empty(size, dtype=object)
    channel_values = df["channel"].to_numpy()
    for channel in ["card_present", "e_commerce", "mobile_app", "atm"]:
        idx = np.where(channel_values == channel)[0]
        submask = fraud_mask[idx]
        if channel == "card_present":
            legit_p, fraud_p = [0.82, 0.16, 0.02], [0.40, 0.45, 0.15]
        elif channel == "e_commerce":
            legit_p, fraud_p = [0.38, 0.45, 0.17], [0.10, 0.38, 0.52]
        elif channel == "mobile_app":
            legit_p, fraud_p = [0.55, 0.34, 0.11], [0.18, 0.42, 0.40]
        else:
            legit_p, fraud_p = [0.76, 0.20, 0.04], [0.30, 0.48, 0.22]
        values = np.empty(len(idx), dtype=object)
        values[~submask] = rng.choice(["low", "medium", "high"], size=(~submask).sum(), p=legit_p)
        values[submask] = rng.choice(["low", "medium", "high"], size=submask.sum(), p=fraud_p)
        device_risk[idx] = values
    df["device_risk_tier"] = device_risk

    # Cross-border behavior
    foreign_country = rng.choice(COUNTRIES, size=size)
    home = df["home_country"].to_numpy().copy()
    foreign_country = np.where(foreign_country == home, np.roll(COUNTRIES, 1)[np.searchsorted(COUNTRIES, home, sorter=np.argsort(COUNTRIES)) % len(COUNTRIES)], foreign_country)
    fraud_domestic = rng.random(size) < np.where(fraud_mask, 0.54, 0.92)
    df["country"] = np.where(fraud_domestic, home, foreign_country)
    df["is_cross_border"] = (df["country"] != df["home_country"]).astype(int)

    probs = {
        "low": [0.82, 0.15, 0.03],
        "medium": [0.62, 0.25, 0.13],
        "high": [0.38, 0.30, 0.32],
    }
    chargeback = np.empty(size, dtype=int)
    for seg, p in probs.items():
        idx = np.where(df["risk_segment"].to_numpy() == seg)[0]
        chargeback[idx] = rng.choice([0, 1, 2], size=len(idx), p=p)
    chargeback[fraud_mask] = np.minimum(chargeback[fraud_mask] + rng.integers(0, 2, size=fraud_mask.sum()), 3)
    df["prior_chargeback_count"] = chargeback

    cat_risk_map = {
        "grocery": 0.08,
        "retail": 0.18,
        "travel": 0.32,
        "electronics": 0.34,
        "digital_goods": 0.48,
        "bill_pay": 0.12,
        "cash_withdrawal": 0.22,
    }
    dev_risk_map = {"low": 0.08, "medium": 0.22, "high": 0.46}
    df["merchant_risk_score"] = (
        df["merchant_category"].map(cat_risk_map).astype(float)
        + df["device_risk_tier"].map(dev_risk_map).astype(float)
    ).clip(0, 1)

    return df



def _derive_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_ts"] = pd.Timestamp(BASELINE_START) + pd.to_timedelta(df["Time"], unit="s")
    df["hour"] = df["event_ts"].dt.hour
    df["day_name"] = df["event_ts"].dt.day_name()
    df["amount_log"] = np.log1p(df["Amount"])
    df["amount_vs_profile"] = (df["Amount"] / df["profile_avg_ticket"]).clip(0, 25)

    df = df.sort_values(["customer_id", "event_ts"]).reset_index(drop=True)
    df["prev_event_ts"] = df.groupby("customer_id")["event_ts"].shift(1)
    df["minutes_since_prev_tx"] = (
        (df["event_ts"] - df["prev_event_ts"]).dt.total_seconds().div(60).fillna(9999).clip(0, 9999)
    )
    df["prev_amount"] = df.groupby("customer_id")["Amount"].shift(1).fillna(df["profile_avg_ticket"])
    df["amount_delta_abs"] = (df["Amount"] - df["prev_amount"]).abs()
    df["customer_tx_index"] = df.groupby("customer_id").cumcount() + 1
    df["customer_recent_fraud_count"] = df.groupby("customer_id")["Class"].cumsum() - df["Class"]

    df = df.sort_values("event_ts").reset_index(drop=True)
    df["rapid_repeat_flag"] = (df["minutes_since_prev_tx"] < 5).astype(int)
    df["new_account_flag"] = (df["account_age_days"] < 45).astype(int)
    df["high_amount_flag"] = (df["amount_vs_profile"] > 5).astype(int)
    df["night_tx_flag"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    df["transaction_id"] = [f"TXN-{i:07d}" for i in range(1, len(df) + 1)]
    return df



def enrich_transactions(raw_df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    enriched = _assign_customers(raw_df, rng)
    enriched = _assign_operational_context(enriched, rng)
    enriched = _derive_temporal_features(enriched)
    return enriched



def build_demo_dataset(enriched_df: pd.DataFrame, legit_rows: int = 60000) -> pd.DataFrame:
    """Create a lighter demo set while preserving every fraud row."""
    fraud_df = enriched_df[enriched_df["Class"] == 1]
    legit_df = enriched_df[enriched_df["Class"] == 0]
    sample_size = max(0, legit_rows - len(fraud_df))
    legit_sample = legit_df.sample(n=min(sample_size, len(legit_df)), random_state=SEED)
    demo_df = (
        pd.concat([fraud_df, legit_sample], axis=0)
        .sort_values("event_ts")
        .reset_index(drop=True)
    )
    return demo_df
