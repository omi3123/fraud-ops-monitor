from pathlib import Path

import pandas as pd

from src.config import DATA_DIR
from src.inference import load_model



def test_model_scores_template() -> None:
    model = load_model()
    template = pd.read_csv(DATA_DIR / "demo_input_template.csv")
    scored = model.score(template)
    assert "alert_priority_score" in scored.columns
    assert len(scored) == len(template)
