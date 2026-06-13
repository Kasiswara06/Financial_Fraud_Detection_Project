import streamlit as st
import pandas as pd

from src.utils import load_dataset
from src.preprocessing import preprocess_data
from src.train_models import (
    train_models,
    save_models
)
from src.evaluation import evaluate_model

st.title("🤖 Model Training")

dataset_name = st.selectbox(
    "Select Dataset",
    [
        "Dataset 1",
        "Dataset 2",
        "Dataset 3"
    ]
)

df, target = load_dataset(
    dataset_name
)

st.write("Target:", target)

if st.button("Train Models"):

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        encoders,
        feature_names
    ) = preprocess_data(
        df,
        target
    )

    st.write(
        "Feature Count:",
        len(feature_names)
    )

    models = train_models(
        X_train,
        y_train
    )

    save_models(
        models,
        dataset_name,
        scaler,
        feature_names
    )

    results = []

    for name, model in models.items():

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        results.append({

            "Model": name,

            "Accuracy":
            round(
                metrics["Accuracy"],
                4
            ),

            "Precision":
            round(
                metrics["Precision"],
                4
            ),

            "Recall":
            round(
                metrics["Recall"],
                4
            ),

            "F1 Score":
            round(
                metrics["F1 Score"],
                4
            ),

            "ROC AUC":
            round(
                metrics["ROC AUC"],
                4
            )
        })

    result_df = pd.DataFrame(
        results
    )

    st.dataframe(
        result_df,
        use_container_width=True
    )

    result_df.to_csv(
        "reports/model_results.csv",
        index=False
    )

    st.success(
        f"{dataset_name} models saved successfully"
    )