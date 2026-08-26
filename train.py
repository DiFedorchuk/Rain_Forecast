from __future__ import annotations

import argparse

from rain_predictor.config import DEFAULT_ARTIFACT_PATH, DEFAULT_DATA_PATH
from rain_predictor.data import load_dataset
from rain_predictor.modeling import load_artifact, retrain_from_dataframe, save_artifact


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain the Aussie rain logistic regression model on a CSV dataset."
    )
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="Path to the weather CSV file.",
    )
    parser.add_argument(
        "--artifact-path",
        default=str(DEFAULT_ARTIFACT_PATH),
        help="Path to the existing model\\aussie_rain.joblib artifact used as the schema source.",
    )
    parser.add_argument(
        "--out-path",
        default=str(DEFAULT_ARTIFACT_PATH),
        help="Output path for the retrained joblib artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_artifact = load_artifact(args.artifact_path)
    dataset = load_dataset(args.data_path)
    new_artifact, metrics = retrain_from_dataframe(dataset, base_artifact)
    save_artifact(new_artifact, args.out_path)

    print(f"Saved retrained artifact to: {args.out_path}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
