from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


RULE_CATALOG = {
    "R1": "Large amount versus customer baseline",
    "R2": "Rapid repeat transaction pattern",
    "R3": "New account with high-value transaction",
    "R4": "Cross-border transaction on elevated merchant/device risk",
    "R5": "Customer has prior chargeback history",
    "R6": "Night-time digital purchase with elevated device risk",
}



def apply_business_rules(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    conditions = {
        "R1": work["amount_vs_profile"] > 5,
        "R2": work["minutes_since_prev_tx"] < 3,
        "R3": (work["account_age_days"] < 45) & (work["Amount"] > 250),
        "R4": (work["is_cross_border"] == 1) & (work["merchant_risk_score"] >= 0.60),
        "R5": work["prior_chargeback_count"] >= 1,
        "R6": (work["night_tx_flag"] == 1)
        & (work["merchant_category"].isin(["digital_goods", "electronics", "travel"]))
        & (work["device_risk_tier"].isin(["medium", "high"])),
    }

    work["rule_hits"] = 0
    work["rule_codes"] = ""
    work["rule_reasons"] = ""

    codes = []
    reasons = []
    hit_counts = np.zeros(len(work), dtype=int)
    for code, mask in conditions.items():
        hit_counts += mask.astype(int)
        codes.append(np.where(mask, code, ""))
        reasons.append(np.where(mask, RULE_CATALOG[code], ""))

    work["rule_hits"] = hit_counts
    code_matrix = np.column_stack(codes)
    reason_matrix = np.column_stack(reasons)

    work["rule_codes"] = [
        ", ".join([v for v in row if v]) for row in code_matrix
    ]
    work["rule_reasons"] = [
        "; ".join([v for v in row if v]) for row in reason_matrix
    ]
    return work



def add_priority_bands(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    bins = [-np.inf, 35, 60, 80, np.inf]
    labels = ["low", "medium", "high", "critical"]
    work["priority_band"] = pd.cut(work["alert_priority_score"], bins=bins, labels=labels)
    work["priority_band"] = work["priority_band"].astype(str)
    actions = {
        "critical": "Block or escalate immediately",
        "high": "Review within 15 minutes",
        "medium": "Queue for analyst review",
        "low": "Monitor only",
    }
    work["recommended_action"] = work["priority_band"].map(actions)
    return work
