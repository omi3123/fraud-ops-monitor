from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEED
from .rules import add_priority_bands, apply_business_rules


@dataclass
class HybridFraudModel:
    preprocessor: ColumnTransformer
    classifier: Any
    isolation_forest: IsolationForest
    threshold: float
    metrics: dict[str, Any]
    numeric_features: list[str]
    categorical_features: list[str]

    def prepare_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        if "amount_log" not in prepared.columns and "Amount" in prepared.columns:
            prepared["amount_log"] = np.log1p(prepared["Amount"].clip(lower=0))
        if "event_ts" in prepared.columns and "hour" not in prepared.columns:
            event_ts = pd.to_datetime(prepared["event_ts"], errors="coerce")
            prepared["hour"] = event_ts.dt.hour.fillna(0).astype(int)
            if "day_name" not in prepared.columns:
                prepared["day_name"] = event_ts.dt.day_name().fillna("Unknown")
        elif "hour" not in prepared.columns:
            prepared["hour"] = 0
        if "day_name" not in prepared.columns:
            prepared["day_name"] = "Unknown"
        if "rapid_repeat_flag" not in prepared.columns and "minutes_since_prev_tx" in prepared.columns:
            prepared["rapid_repeat_flag"] = (prepared["minutes_since_prev_tx"] < 5).astype(int)
        if "new_account_flag" not in prepared.columns and "account_age_days" in prepared.columns:
            prepared["new_account_flag"] = (prepared["account_age_days"] < 45).astype(int)
        if "high_amount_flag" not in prepared.columns and "amount_vs_profile" in prepared.columns:
            prepared["high_amount_flag"] = (prepared["amount_vs_profile"] > 5).astype(int)
        if "night_tx_flag" not in prepared.columns and "hour" in prepared.columns:
            prepared["night_tx_flag"] = prepared["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
        for col in self.numeric_features + self.categorical_features:
            if col not in prepared.columns:
                prepared[col] = np.nan if col in self.numeric_features else "Unknown"
        return prepared

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        prepared = self.prepare_features(frame)
        return self.preprocessor.transform(prepared[self.numeric_features + self.categorical_features])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.transform(frame)
        return self.classifier.predict_proba(transformed)[:, 1]

    def anomaly_score(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.transform(frame)
        raw = -self.isolation_forest.score_samples(transformed)
        min_v = raw.min()
        max_v = raw.max()
        if max_v == min_v:
            return np.zeros_like(raw)
        return (raw - min_v) / (max_v - min_v)

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        scored = self.prepare_features(frame)
        scored = apply_business_rules(scored)
        scored["fraud_probability"] = self.predict_proba(scored)
        scored["anomaly_score"] = self.anomaly_score(scored)
        amount_component = np.clip(np.log1p(scored["Amount"]) / np.log(1000), 0, 1)
        rule_component = np.clip(scored["rule_hits"] / 4, 0, 1)
        scored["alert_priority_score"] = (
            100
            * (
                0.55 * scored["fraud_probability"]
                + 0.20 * scored["anomaly_score"]
                + 0.15 * amount_component
                + 0.10 * rule_component
            )
        ).round(2)
        scored["predicted_fraud"] = (scored["fraud_probability"] >= self.threshold).astype(int)
        scored["needs_review"] = (scored["alert_priority_score"] >= 60).astype(int)
        scored = add_priority_bands(scored)
        scored["risk_summary"] = scored.apply(_row_summary, axis=1)
        return scored

    def save(self, output_dir: Path | str) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, output_dir / "preprocessor.joblib")
        joblib.dump(self.classifier, output_dir / "fraud_hybrid_model.joblib")
        joblib.dump(self.isolation_forest, output_dir / "isolation_forest.joblib")
        (output_dir / "metrics.json").write_text(json.dumps(self.metrics, indent=2), encoding="utf-8")
        (output_dir / "feature_schema.json").write_text(
            json.dumps(
                {
                    "numeric_features": self.numeric_features,
                    "categorical_features": self.categorical_features,
                    "threshold": self.threshold,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, artifact_dir: Path | str) -> "HybridFraudModel":
        artifact_dir = Path(artifact_dir)
        preprocessor = joblib.load(artifact_dir / "preprocessor.joblib")
        classifier = joblib.load(artifact_dir / "fraud_hybrid_model.joblib")
        isolation_forest = joblib.load(artifact_dir / "isolation_forest.joblib")
        metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
        schema = json.loads((artifact_dir / "feature_schema.json").read_text(encoding="utf-8"))
        return cls(
            preprocessor=preprocessor,
            classifier=classifier,
            isolation_forest=isolation_forest,
            threshold=float(schema["threshold"]),
            metrics=metrics,
            numeric_features=schema["numeric_features"],
            categorical_features=schema["categorical_features"],
        )



def _row_summary(row: pd.Series) -> str:
    reasons = []
    if row.get("fraud_probability", 0) >= 0.85:
        reasons.append("very high model fraud probability")
    if row.get("anomaly_score", 0) >= 0.75:
        reasons.append("unusual anomaly pattern")
    if row.get("rule_reasons"):
        reasons.append(row["rule_reasons"])
    if not reasons:
        reasons.append("monitoring score elevated without a single dominant trigger")
    return "; ".join(reasons[:3])



def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )



def fit_hybrid_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> HybridFraudModel:
    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df["Class"].astype(int)
    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df["Class"].astype(int)

    preprocessor = _build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    pos = max(1, int(y_train.sum()))
    neg = max(1, int((1 - y_train).sum()))
    scale_pos_weight = neg / pos

    classifier = XGBClassifier(
        n_estimators=260,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=SEED,
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
    )
    classifier.fit(X_train_t, y_train)

    train_majority = train_df[train_df["Class"] == 0].sample(n=min(50000, int((train_df["Class"] == 0).sum())), random_state=SEED)
    iso_input = preprocessor.transform(train_majority[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    isolation_forest = IsolationForest(
        n_estimators=180,
        contamination=0.01,
        random_state=SEED,
        n_jobs=4,
    )
    isolation_forest.fit(iso_input)

    probabilities = classifier.predict_proba(X_test_t)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, probabilities)
    f1 = (2 * precision * recall) / np.clip(precision + recall, 1e-9, None)
    best_idx = int(np.nanargmax(f1[:-1])) if len(thresholds) > 0 else 0
    threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "chosen_threshold": threshold,
        "precision_at_threshold": float(precision[best_idx]),
        "recall_at_threshold": float(recall[best_idx]),
        "test_positive_rate": float(y_test.mean()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }

    return HybridFraudModel(
        preprocessor=preprocessor,
        classifier=classifier,
        isolation_forest=isolation_forest,
        threshold=threshold,
        metrics=metrics,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
