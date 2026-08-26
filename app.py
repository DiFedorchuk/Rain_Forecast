from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from rain_predictor.config import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_IMAGE_PATH,
)
from rain_predictor.data import load_dataset
from rain_predictor.modeling import (
    load_artifact,
    predict_with_probability,
    preprocess_user_input,
)
from rain_predictor.ui import build_input_form


@st.cache_resource
def get_artifact() -> dict:
    return load_artifact(DEFAULT_ARTIFACT_PATH)


@st.cache_data
def get_reference_data() -> pd.DataFrame | None:
    if not DEFAULT_DATA_PATH.exists():
        return None
    return load_dataset(DEFAULT_DATA_PATH)


def main() -> None:
    st.set_page_config(page_title="Australian Rain Predictor", page_icon="🌦️", layout="wide")
    st.title("Australian Rain Predictor")

    if DEFAULT_IMAGE_PATH.exists():
        st.image(str(DEFAULT_IMAGE_PATH), use_container_width=True)

    if not DEFAULT_ARTIFACT_PATH.exists():
        st.error(
            "Missing model\\aussie_rain.joblib. Retrain the model or copy the "
            "artifact into the model folder before running the app"
        )
        st.stop()

    artifact = get_artifact()
    reference_df = get_reference_data()

    with st.sidebar:
        st.header("Project overview")
        st.write(
            "This app uses a saved scikit-learn pipeline artifact built from the "
            "Australian weather dataset"
        )
        st.metric("Input features", len(artifact["input_cols"]))
        st.metric("Encoded features", len(artifact["encoded_cols"]))
        if reference_df is not None:
            st.metric("Reference rows", len(reference_df))

    intro_col, preview_col = st.columns((3, 2))
    with intro_col:
        st.subheader("Forecast form")
        st.write(
            "Fill in the weather conditions below. Numeric fields are pre-filled "
            "with values learned from the trained preprocessing pipeline"
        )
    with preview_col:
        if reference_df is not None:
            with st.expander("Preview training data"):
                st.dataframe(reference_df.head(), use_container_width=True)
        else:
            st.info("Dataset preview is unavailable in deployment-only mode")

    with st.form("forecast-form"):
        input_data = build_input_form(artifact)
        submitted = st.form_submit_button("Forecast rain", type="primary")

    if submitted:
        processed_input = preprocess_user_input(input_data, artifact)
        prediction, probability = predict_with_probability(processed_input, artifact)
        label = "Yes" if prediction == "Yes" else "No"
        st.success(f"Rain tomorrow: {label}")
        st.metric("Forecast probability", f"{probability:.2%}")
        st.caption(
            "Data flow: input -> imputation -> scaling -> one-hot encoding -> "
            "logistic regression prediction"
        )


if __name__ == "__main__":
    # Guard against launching with `python app.py` instead of `streamlit run app.py`.
    if get_script_run_ctx() is None:
        print("This is a Streamlit app.")
        print("Run it with: streamlit run app.py")
    else:
        main()
