import joblib
import pandas as pd
import os


def predict_transaction(
        input_data,
        dataset_name,
        model_name
):

    prefix = dataset_name.replace(
        " ",
        "_"
    )

    model_paths = {

        "Logistic Regression":
        f"models/{prefix}_logistic.pkl",

        "Decision Tree":
        f"models/{prefix}_tree.pkl",

        "Random Forest":
        f"models/{prefix}_rf.pkl"
    }

    # Check model exists

    if not os.path.exists(
        model_paths[model_name]
    ):

        raise FileNotFoundError(
            f"""
Model not found:

{model_paths[model_name]}

Please train models for
{dataset_name} first.
"""
        )

    feature_file = (
        f"models/{prefix}_features.pkl"
    )

    scaler_file = (
        f"models/{prefix}_scaler.pkl"
    )

    if not os.path.exists(
        feature_file
    ):

        raise FileNotFoundError(
            f"""
Feature file not found:

{feature_file}

Please retrain
{dataset_name}.
"""
        )

    if not os.path.exists(
        scaler_file
    ):

        raise FileNotFoundError(
            f"""
Scaler file not found:

{scaler_file}

Please retrain
{dataset_name}.
"""
        )

    # Load artifacts

    model = joblib.load(
        model_paths[model_name]
    )

    feature_names = joblib.load(
        feature_file
    )

    scaler = joblib.load(
        scaler_file
    )

    # Create dataframe

    input_df = pd.DataFrame(
        [input_data]
    )

    # Add missing columns

    for col in feature_names:

        if col not in input_df.columns:

            input_df[col] = 0

    # Remove unwanted columns

    input_df = input_df[
        feature_names
    ]

    # Convert to numeric

    input_df = input_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    input_df = input_df.fillna(
        0
    )

    # Scale

    input_scaled = scaler.transform(
        input_df
    )

    # Prediction

    prediction = model.predict(
        input_scaled
    )[0]

    try:

        probability = (
            model.predict_proba(
                input_scaled
            )[0][1]
        )

    except:

        probability = 0.0

    return prediction, probability