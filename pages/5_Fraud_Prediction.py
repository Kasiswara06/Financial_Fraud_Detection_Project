import streamlit as st
import pandas as pd
import joblib
import os

from src.utils import load_dataset
from src.prediction import predict_transaction

st.title("🚨 Financial Fraud Prediction")

# ==================================================
# DATASET SELECTION
# ==================================================

dataset_name = st.selectbox(
    "Select Dataset",
    [
        "Dataset 1",
        "Dataset 2",
        "Dataset 3"
    ]
)

try:

    df, target = load_dataset(
        dataset_name
    )

    st.success(
        f"Target Column Detected: {target}"
    )

    prefix = dataset_name.replace(
        " ",
        "_"
    )

    feature_file = (
        f"models/{prefix}_features.pkl"
    )

    if not os.path.exists(
        feature_file
    ):

        st.error(
            f"""
Feature file not found:

{feature_file}

Please train models first.
"""
        )

        st.stop()

    # ==================================================
    # LOAD FEATURES
    # ==================================================

    feature_names = joblib.load(
        feature_file
    )

    st.subheader(
        "Enter Transaction Details"
    )

    user_input = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(
        feature_names
    ):

        default_value = 0.0

        if feature in df.columns:

            try:

                default_value = float(
                    pd.to_numeric(
                        df[feature],
                        errors="coerce"
                    ).mean()
                )

            except:

                default_value = 0.0

        if i % 2 == 0:

            with col1:

                user_input[
                    feature
                ] = st.number_input(
                    feature,
                    value=float(
                        default_value
                    )
                )

        else:

            with col2:

                user_input[
                    feature
                ] = st.number_input(
                    feature,
                    value=float(
                        default_value
                    )
                )

    # ==================================================
    # MODEL SELECTION
    # ==================================================

    model_name = st.selectbox(
        "Choose Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ]
    )

    # ==================================================
    # PREDICTION
    # ==================================================

    if st.button(
        "Predict Fraud"
    ):

        prediction, probability = (
            predict_transaction(
                user_input,
                dataset_name,
                model_name
            )
        )

        st.subheader(
            "Prediction Result"
        )

        st.metric(
            "Fraud Probability",
            f"{probability:.2%}"
        )

        if prediction == 1:

            st.error(
                "🚨 Fraudulent Transaction Detected"
            )

        else:

            st.success(
                "✅ Legitimate Transaction"
            )

except FileNotFoundError as e:

    st.error(
        f"File Error: {e}"
    )

except Exception as e:

    st.error(
        f"Error: {e}"
    )