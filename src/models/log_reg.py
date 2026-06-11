from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from typing import Any


def train_log_reg(X_train, X_test, y_train, numeric_features, categorical_features=None):
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

    model = LogisticRegression(
        max_iter=150,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
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

    y_proba = pipeline.predict_proba(
        X_test
    )[:, 1]

    y_pred = pipeline.predict(
        X_test
    )

    coefficients = (
        pipeline.named_steps["model"]
        .coef_[0]
    )

    feature_names = (
        pipeline.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    print('log reg convergence', pipeline.named_steps["model"].n_iter_)

    return (
        y_proba,
        y_pred,
        coefficients,
        feature_names
    )
