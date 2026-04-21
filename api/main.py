from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.config import ARTIFACT_DIR
from src.inference import load_model


app = FastAPI(title="Fraud Ops Monitor API", version="1.0.0")
model = load_model(ARTIFACT_DIR)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_threshold": model.threshold,
        "metrics": model.metrics,
    }


@app.post("/score")
async def score_csv(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    content = await file.read()
    try:
        frame = pd.read_csv(pd.io.common.BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Unable to read CSV: {exc}") from exc

    required = set(model.numeric_features + model.categorical_features)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HTTPException(status_code=400, detail={"missing_columns": missing})

    scored = model.score(frame)
    output_cols = [
        col for col in [
            "transaction_id", "Amount", "channel", "merchant_category", "country", "priority_band",
            "alert_priority_score", "fraud_probability", "anomaly_score", "rule_hits", "risk_summary"
        ] if col in scored.columns
    ]
    return JSONResponse(content={"rows": scored[output_cols].to_dict(orient="records")})
