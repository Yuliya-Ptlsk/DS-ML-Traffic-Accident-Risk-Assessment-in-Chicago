from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.neural_network import MLPClassifier
from typing import Any

def train_mlp(
    X_train,
    y_train,
    X_test,
    numeric_features,
    categorical_features=None,
):
    transformers: list[tuple[str, object, Any]] = [
        (
            "num",
            StandardScaler(),
            numeric_features,
        )
    ]

    if categorical_features:
        transformers.append(
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=True,
    )

    pipeline = Pipeline(
        [
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    y_proba = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    y_pred = pipeline.predict(
        X_test
    )

    # y_pred = (
    #     y_proba > 0.10
    # ).astype(int)

    return (
        y_proba,
        y_pred,
        pipeline,
    )