import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def train_models(X_train, y_train):

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
    }

    trained_models = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        trained_models[name] = model

    return trained_models


def save_models(
    models,
    dataset_name,
    scaler,
    feature_names
):

    os.makedirs(
        "models",
        exist_ok=True
    )

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


def load_models(dataset_name):

    prefix = dataset_name.replace(
        " ",
        "_"
    )

    logistic = joblib.load(
        f"models/{prefix}_logistic.pkl"
    )

    tree = joblib.load(
        f"models/{prefix}_tree.pkl"
    )

    rf = joblib.load(
        f"models/{prefix}_rf.pkl"
    )

    scaler = joblib.load(
        f"models/{prefix}_scaler.pkl"
    )

    feature_names = joblib.load(
        f"models/{prefix}_features.pkl"
    )

    models = {

        "Logistic Regression": logistic,

        "Decision Tree": tree,

        "Random Forest": rf
    }

    return (
        models,
        scaler,
        feature_names
    )


def model_exists(dataset_name):

    prefix = dataset_name.replace(
        " ",
        "_"
    )

    required_files = [

        f"models/{prefix}_logistic.pkl",

        f"models/{prefix}_tree.pkl",

        f"models/{prefix}_rf.pkl",

        f"models/{prefix}_scaler.pkl",

        f"models/{prefix}_features.pkl"
    ]

    return all(
        os.path.exists(file)
        for file in required_files
    )