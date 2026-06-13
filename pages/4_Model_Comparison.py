import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix
)

from src.utils import load_dataset
from src.preprocessing import preprocess_data

st.title("📈 Model Comparison")

# Dataset Selection
dataset_name = st.selectbox(
    "Select Dataset",
    [
        "Dataset 1",
        "Dataset 2",
        "Dataset 3"
    ]
)

try:

    # Load Dataset
    df, target = load_dataset(
        dataset_name
    )

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

    prefix = dataset_name.replace(
        " ",
        "_"
    )

    # Load Dataset-Specific Models
    models = {

        "Logistic Regression":
        joblib.load(
            f"models/{prefix}_logistic.pkl"
        ),

        "Decision Tree":
        joblib.load(
            f"models/{prefix}_tree.pkl"
        ),

        "Random Forest":
        joblib.load(
            f"models/{prefix}_rf.pkl"
        )
    }

    st.success(
        f"Loaded Models For {dataset_name}"
    )

    st.write(
        "Current Dataset Features:",
        X_test.shape[1]
    )

    # ROC Curve Section

    st.subheader(
        "ROC Curve Comparison"
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for name, model in models.items():

        try:

            st.write(
                f"{name} expects:",
                model.n_features_in_,
                "features"
            )

            if (
                model.n_features_in_
                != X_test.shape[1]
            ):

                st.warning(
                    f"""
{name}

Expected Features:
{model.n_features_in_}

Current Dataset Features:
{X_test.shape[1]}

Retrain models for this dataset.
"""
                )

                continue

            probs = model.predict_proba(
                X_test
            )[:, 1]

            fpr, tpr, _ = roc_curve(
                y_test,
                probs
            )

            auc = roc_auc_score(
                y_test,
                probs
            )

            ax.plot(
                fpr,
                tpr,
                label=f"{name} (AUC={auc:.3f})"
            )

        except Exception as e:

            st.error(
                f"{name}: {str(e)}"
            )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve Comparison"
    )

    ax.legend()

    st.pyplot(fig)

    # Confusion Matrix Section

    st.subheader(
        "Confusion Matrices"
    )

    for name, model in models.items():

        try:

            if (
                model.n_features_in_
                != X_test.shape[1]
            ):
                continue

            predictions = model.predict(
                X_test
            )

            cm = confusion_matrix(
                y_test,
                predictions
            )

            st.write(
                f"### {name}"
            )

            fig2, ax2 = plt.subplots(
                figsize=(5, 4)
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax2
            )

            st.pyplot(fig2)

        except Exception as e:

            st.error(
                f"{name}: {str(e)}"
            )

except FileNotFoundError:

    st.error(
        """
Models not found.

Please go to:
Model Training

Select this dataset and train models first.
"""
    )

except Exception as e:

    st.error(
        f"Error: {str(e)}"
    )