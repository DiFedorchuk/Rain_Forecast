from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from rain_predictor.config import DEFAULT_DATA_PATH, TARGET_COL


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_modeling_dataframe(
    raw_df: pd.DataFrame, target_col: str = TARGET_COL
) -> pd.DataFrame:
    required_cols = ["RainToday", target_col]
    missing = [column for column in required_cols if column not in raw_df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for modeling: {missing}. "
            f"Available columns: {list(raw_df.columns)}"
        )
    return raw_df.dropna(subset=required_cols).copy()


def split_dataset(
    modeling_df: pd.DataFrame,
    target_col: str = TARGET_COL,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val_df, test_df = train_test_split(
        modeling_df,
        test_size=0.2,
        random_state=random_state,
        stratify=modeling_df[target_col],
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.25,
        random_state=random_state,
        stratify=train_val_df[target_col],
    )
    return train_df, val_df, test_df
