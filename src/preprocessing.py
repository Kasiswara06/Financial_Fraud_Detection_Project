import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(df, target_column):

    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Drop completely empty columns
    df = df.dropna(axis=1, how="all")

    # Fill missing values
    for col in df.columns:

        if col == target_column:
            continue

        # Numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):

            df[col] = df[col].fillna(
                df[col].median()
            )

        # Categorical / String columns
        else:

            mode = df[col].mode()

            if len(mode) > 0:
                df[col] = df[col].fillna(mode[0])
            else:
                df[col] = df[col].fillna("Unknown")

    # Encode categorical columns
    encoders = {}

    for col in df.columns:

        if col == target_column:
            continue

        if (
            df[col].dtype == "object"
            or str(df[col].dtype).startswith("string")
            or pd.api.types.is_string_dtype(df[col])
        ):

            le = LabelEncoder()

            df[col] = le.fit_transform(
                df[col].astype(str)
            )

            encoders[col] = le

    # Encode target if needed
    if (
        df[target_column].dtype == "object"
        or str(df[target_column].dtype).startswith("string")
        or pd.api.types.is_string_dtype(df[target_column])
    ):

        le = LabelEncoder()

        df[target_column] = le.fit_transform(
            df[target_column].astype(str)
        )

    X = df.drop(columns=[target_column])

    y = df[target_column]

    feature_names = list(X.columns)

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        encoders,
        feature_names
    )