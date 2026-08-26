from __future__ import annotations

from typing import Any

import streamlit as st

from rain_predictor.modeling import get_numeric_defaults


Artifact = dict[str, Any]


def build_input_form(artifact: Artifact) -> dict[str, float | str]:
    values: dict[str, float | str] = {}
    input_cols = list(artifact["input_cols"])
    numeric_cols = list(artifact["numeric_cols"])
    categorical_cols = list(artifact["categorical_cols"])
    numeric_defaults = get_numeric_defaults(artifact)
    encoder = artifact["encoder"]

    left_col, right_col = st.columns(2)
    for index, column in enumerate(input_cols):
        target_col = left_col if index % 2 == 0 else right_col
        with target_col:
            if column in numeric_cols:
                values[column] = st.number_input(
                    column,
                    value=float(numeric_defaults.get(column, 0.0)),
                    step=0.1,
                    format="%.3f",
                )
                continue

            category_index = categorical_cols.index(column)
            options = [str(value) for value in encoder.categories_[category_index]]
            values[column] = st.selectbox(column, options=options, index=0)

    return values
