import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def train_models(X_train, y_train):

    models = {
        "Logistic Regression":
        LogisticRegression(max_iter=1000),

        "Decision Tree":
        DecisionTreeClassifier(random_state=42),

        "Random Forest":
        RRandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    }

    trained_models = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models


def save_models(
        models,
        dataset_name,
        scaler,
        feature_names
):

    os.makedirs("models", exist_ok=True)

    prefix = dataset_name.replace(
        " ",
        "_"
    )

    joblib.dump(
        models["Logistic Regression"],
        f"models/{prefix}_logistic.pkl"
    )

    joblib.dump(
        models["Decision Tree"],
        f"models/{prefix}_tree.pkl"
    )

    joblib.dump(
        models["Random Forest"],
        f"models/{prefix}_rf.pkl"
    )

    joblib.dump(
        scaler,
        f"models/{prefix}_scaler.pkl"
    )

    joblib.dump(
        feature_names,
        f"models/{prefix}_features.pkl"
    )