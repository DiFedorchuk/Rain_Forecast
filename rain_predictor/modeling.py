from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from rain_predictor.config import DEFAULT_ARTIFACT_PATH, TARGET_COL
from rain_predictor.data import prepare_modeling_dataframe, split_dataset


Artifact = dict[str, Any]


def patch_imputer(imputer: Any) -> Any:
    if imputer is not None and not hasattr(imputer, "_fill_dtype"):
        stats = getattr(imputer, "statistics_", None)
        dtype = stats.dtype if stats is not None and hasattr(stats, "dtype") else np.float64
        object.__setattr__(imputer, "_fill_dtype", dtype)
    return imputer


def patch_artifact_compatibility(artifact: Artifact) -> Artifact:
    imputer = artifact.get("imputer")
    if imputer is not None:
        patch_imputer(imputer)
    return artifact


def load_artifact(artifact_path: str | Path = DEFAULT_ARTIFACT_PATH) -> Artifact:
    artifact = joblib.load(artifact_path)
    return patch_artifact_compatibility(artifact)


def save_artifact(
    artifact: Artifact, out_path: str | Path = DEFAULT_ARTIFACT_PATH
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)


def validate_required_columns(
    df: pd.DataFrame, cols: list[str], context: str
) -> None:
    missing = [column for column in cols if column not in df.columns]
    if missing:
        raise ValueError(
            f"{context}: missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def transform_inputs(
    inputs_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    encoded_cols: list[str],
    imputer: SimpleImputer,
    scaler: MinMaxScaler,
    encoder: OneHotEncoder,
) -> pd.DataFrame:
    patch_imputer(imputer)
    imputed_numeric = pd.DataFrame(
        imputer.transform(inputs_df[numeric_cols]),
        columns=numeric_cols,
        index=inputs_df.index,
    )
    scaled_numeric = pd.DataFrame(
        scaler.transform(imputed_numeric),
        columns=numeric_cols,
        index=inputs_df.index,
    )
    encoded_categorical = pd.DataFrame(
        encoder.transform(inputs_df[categorical_cols]),
        columns=encoded_cols,
        index=inputs_df.index,
    )
    return pd.concat([scaled_numeric, encoded_categorical], axis=1)


def preprocess_user_input(user_input: dict, artifact: Artifact) -> pd.DataFrame:
    patch_artifact_compatibility(artifact)
    input_cols = list(artifact["input_cols"])
    input_df = pd.DataFrame([user_input])
    validate_required_columns(input_df, input_cols, "Prediction input")
    ordered_input_df = input_df[input_cols].copy()

    return transform_inputs(
        ordered_input_df,
        list(artifact["numeric_cols"]),
        list(artifact["categorical_cols"]),
        list(artifact["encoded_cols"]),
        artifact["imputer"],
        artifact["scaler"],
        artifact["encoder"],
    )


def get_numeric_defaults(artifact: Artifact) -> dict[str, float]:
    patch_artifact_compatibility(artifact)
    numeric_cols = list(artifact["numeric_cols"])
    stored_defaults = artifact.get("numeric_defaults")
    if isinstance(stored_defaults, dict):
        return {
            column: float(stored_defaults.get(column, 0.0)) for column in numeric_cols
        }

    imputer = artifact["imputer"]
    return {
        column: float(value)
        for column, value in zip(numeric_cols, imputer.statistics_)
    }


def predict_with_probability(
    processed_input: pd.DataFrame, artifact: Artifact
) -> tuple[str, float]:
    model = artifact["model"]
    classes = list(model.classes_)
    probabilities = model.predict_proba(processed_input)[0]

    best_idx = int(np.argmax(probabilities))
    predicted_label = str(classes[best_idx])
    predicted_probability = float(probabilities[best_idx])
    return predicted_label, predicted_probability


def predict_input(single_input: dict, artifact: Artifact) -> tuple[str, float]:
    processed_input = preprocess_user_input(single_input, artifact)
    return predict_with_probability(processed_input, artifact)


def retrain_from_dataframe(
    new_df: pd.DataFrame,
    base_artifact: Artifact,
    random_state: int = 42,
) -> tuple[Artifact, dict[str, float | int]]:
    input_cols = list(base_artifact["input_cols"])
    target_col = str(base_artifact.get("target_col", TARGET_COL))
    numeric_cols = list(base_artifact["numeric_cols"])
    categorical_cols = list(base_artifact["categorical_cols"])

    validate_required_columns(new_df, input_cols + [target_col], "New training data")

    modeling_df = prepare_modeling_dataframe(new_df, target_col=target_col)
    if len(modeling_df) < 100:
        raise ValueError(
            "Not enough usable rows after dropping rows with missing labels: "
            f"{len(modeling_df)}"
        )

    train_df, val_df, test_df = split_dataset(
        modeling_df,
        target_col=target_col,
        random_state=random_state,
    )

    train_inputs = train_df[input_cols].copy()
    val_inputs = val_df[input_cols].copy()
    test_inputs = test_df[input_cols].copy()

    train_targets = train_df[target_col].copy()
    val_targets = val_df[target_col].copy()
    test_targets = test_df[target_col].copy()

    imputer = SimpleImputer(strategy="mean").fit(train_inputs[numeric_cols])
    imputed_train_numeric = pd.DataFrame(
        imputer.transform(train_inputs[numeric_cols]),
        columns=numeric_cols,
        index=train_inputs.index,
    )
    scaler = MinMaxScaler().fit(imputed_train_numeric)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
        train_inputs[categorical_cols]
    )
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    x_train = transform_inputs(
        train_inputs,
        numeric_cols,
        categorical_cols,
        encoded_cols,
        imputer,
        scaler,
        encoder,
    )
    x_val = transform_inputs(
        val_inputs,
        numeric_cols,
        categorical_cols,
        encoded_cols,
        imputer,
        scaler,
        encoder,
    )
    x_test = transform_inputs(
        test_inputs,
        numeric_cols,
        categorical_cols,
        encoded_cols,
        imputer,
        scaler,
        encoder,
    )

    model = LogisticRegression(solver="liblinear", max_iter=1000)
    model.fit(x_train, train_targets)

    train_preds = model.predict(x_train)
    val_preds = model.predict(x_val)
    test_preds = model.predict(x_test)

    metrics = {
        "rows_used": int(len(modeling_df)),
        "train_accuracy": float(accuracy_score(train_targets, train_preds)),
        "val_accuracy": float(accuracy_score(val_targets, val_preds)),
        "test_accuracy": float(accuracy_score(test_targets, test_preds)),
        "train_f1_yes": float(f1_score(train_targets, train_preds, pos_label="Yes")),
        "val_f1_yes": float(f1_score(val_targets, val_preds, pos_label="Yes")),
        "test_f1_yes": float(f1_score(test_targets, test_preds, pos_label="Yes")),
    }

    artifact = {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "encoder": encoder,
        "input_cols": input_cols,
        "target_col": target_col,
        "numeric_cols": numeric_cols,
        "numeric_defaults": {
            column: float(value)
            for column, value in zip(numeric_cols, imputer.statistics_)
        },
        "categorical_cols": categorical_cols,
        "encoded_cols": encoded_cols,
    }
    return artifact, metrics
