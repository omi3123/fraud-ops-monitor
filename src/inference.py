from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ARTIFACT_DIR
from .modeling import HybridFraudModel



def load_model(artifact_dir: Path | str = ARTIFACT_DIR) -> HybridFraudModel:
    return HybridFraudModel.load(artifact_dir)



def score_frame(frame: pd.DataFrame, artifact_dir: Path | str = ARTIFACT_DIR) -> pd.DataFrame:
    model = load_model(artifact_dir)
    return model.score(frame)
