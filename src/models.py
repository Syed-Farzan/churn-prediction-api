from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.preprocessing import create_preprocessor


def create_models(numeric_features, categorical_features):
    preprocessor = create_preprocessor(numeric_features, categorical_features)

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor(numeric_features, categorical_features),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "svm": Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor(numeric_features, categorical_features),
                ),
                (
                    "classifier",
                    SVC(
                        probability=True,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    return models
