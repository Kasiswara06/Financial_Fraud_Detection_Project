import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.model_selection import (
    train_test_split
)


def preprocess_data(df, target_column):

    # ============================
    # Validation
    # ============================

    if target_column not in df.columns:

        raise ValueError(
            f"""
Target column '{target_column}' not found.

Available Columns:

{list(df.columns)}
"""
        )

    df = df.copy()

    # ============================
    # Remove Duplicate Rows
    # ============================

    df = df.drop_duplicates()

    # ============================
    # Handle Missing Values
    # ============================

    for col in df.columns:

        if df[col].dtype == "object":

            df[col] = df[col].fillna(
                "Unknown"
            )

        else:

            df[col] = df[col].fillna(
                df[col].median()
            )

    # ============================
    # Convert Date Columns
    # ============================

    for col in df.columns:

        try:

            if (
                "date" in col.lower()
                or "time" in col.lower()
            ):

                df[col] = pd.to_datetime(
                    df[col],
                    errors="coerce"
                )

                df[f"{col}_year"] = (
                    df[col].dt.year
                )

                df[f"{col}_month"] = (
                    df[col].dt.month
                )

                df[f"{col}_day"] = (
                    df[col].dt.day
                )

                df.drop(
                    columns=[col],
                    inplace=True
                )

        except:
            pass

    # ============================
    # Encode Target
    # ============================

    if df[target_column].dtype == "object":

        target_encoder = LabelEncoder()

        df[target_column] = (
            target_encoder.fit_transform(
                df[target_column]
            )
        )

    # ============================
    # Encode Features
    # ============================

    feature_encoders = {}

    feature_columns = [

        col for col in df.columns

        if col != target_column
    ]

    for col in feature_columns:

        if df[col].dtype == "object":

            encoder = LabelEncoder()

            df[col] = encoder.fit_transform(
                df[col].astype(str)
            )

            feature_encoders[col] = encoder

    # ============================
    # Features & Target
    # ============================

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    # ============================
    # Convert to Numeric
    # ============================

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.fillna(0)

    # ============================
    # Train Test Split
    # ============================

    try:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )
        )

    except:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42
            )
        )

    # ============================
    # Scaling
    # ============================

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    # ============================
    # Return
    # ============================

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        feature_encoders,
        X.columns.tolist()
    )