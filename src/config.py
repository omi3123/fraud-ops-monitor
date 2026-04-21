from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ASSET_DIR = PROJECT_ROOT / "assets"

RAW_DATA_URL = "https://github.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/raw/refs/heads/master/creditcard.csv"
RAW_DATA_PATH = DATA_DIR / "creditcard.csv"
ENRICHED_DATA_PATH = DATA_DIR / "transactions_enriched.csv"
DEMO_DATA_PATH = DATA_DIR / "demo_scored_transactions.csv"
MODEL_PATH = ARTIFACT_DIR / "fraud_hybrid_model.joblib"
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.joblib"
ISOLATION_PATH = ARTIFACT_DIR / "isolation_forest.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
FEATURE_SCHEMA_PATH = ARTIFACT_DIR / "feature_schema.json"
CASE_QUEUE_PATH = ARTIFACT_DIR / "alert_queue.csv"

SEED = 42
BASELINE_START = "2025-01-01 00:00:00"
DEFAULT_ALERT_BUDGET_RATE = 0.01

NUMERIC_FEATURES = [
    *[f"V{i}" for i in range(1, 29)],
    "Time",
    "Amount",
    "amount_log",
    "account_age_days",
    "amount_vs_profile",
    "minutes_since_prev_tx",
    "amount_delta_abs",
    "customer_tx_index",
    "customer_recent_fraud_count",
    "prior_chargeback_count",
    "merchant_risk_score",
    "is_cross_border",
    "rapid_repeat_flag",
    "new_account_flag",
    "high_amount_flag",
    "night_tx_flag",
    "hour",
]

CATEGORICAL_FEATURES = [
    "channel",
    "merchant_category",
    "device_risk_tier",
    "risk_segment",
    "country",
    "home_country",
    "day_name",
]

AUXILIARY_COLUMNS = [
    "transaction_id",
    "event_ts",
    "customer_id",
    "home_country",
    "country",
    "channel",
    "merchant_category",
    "device_risk_tier",
    "risk_segment",
    "Amount",
    "Class",
]
