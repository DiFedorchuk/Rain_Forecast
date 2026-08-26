from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
IMAGE_DIR = PROJECT_ROOT / "image"
MODEL_DIR = PROJECT_ROOT / "model"

DEFAULT_DATA_PATH = DATA_DIR / "weatherAUS.csv"
DEFAULT_ARTIFACT_PATH = MODEL_DIR / "aussie_rain.joblib"
DEFAULT_IMAGE_PATH = IMAGE_DIR / "weather.jpg"
TARGET_COL = "RainTomorrow"
