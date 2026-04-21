from pathlib import Path

from src.config import RAW_DATA_PATH, RAW_DATA_URL
from src.data_prep import download_raw_data


if __name__ == "__main__":
    path = download_raw_data(RAW_DATA_PATH, RAW_DATA_URL)
    print(f"Saved raw dataset to {Path(path).resolve()}")
