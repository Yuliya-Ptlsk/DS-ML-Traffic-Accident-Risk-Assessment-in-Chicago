import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.catboost_classifier import train_catboost
from src.models.metrics import evaluate_model, feature_importance, save_model_metrics
from src.models.threshold import find_best_threshold
from src.utils.conf_matrix import save_confusion_matrix
from src.visualization.plots import (
    plot_roc_curve,
    plot_feature_importance,
    plot_threshold_metrics,
    plot_confusion_matrix,
)
from pathlib import Path

df = pd.read_parquet("../../data/processed/traffic_crashes_hourly.parquet")

FEATURES_NUMERIC = [
    "avg_speed",
    "avg_free_flow_speed",
    "avg_congestion",
    "segment_lat",
    "segment_lon",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

FEATURES_CATEGORICAL = [
    "segment_id",
    "hour",
    "day_of_week",
    "month",
]

FEATURES = FEATURES_NUMERIC + FEATURES_CATEGORICAL

TARGET = "accident"

X = df[FEATURES]

y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

# ---------------------------------------------
# TRAIN CATBOOST MODEL
# ---------------------------------------------
catboost = train_catboost(X_train, y_train, X_test, y_test, FEATURES_CATEGORICAL)

(
    model,
    y_proba,
    _,
    importance_pred_val,
    importance_loss,
    shap_values
) = catboost

best_threshold_catboost, threshold_results_catboost = (
    find_best_threshold(
        y_test,
        y_proba
    )
)

print(
    f"CatBoost best threshold: {best_threshold_catboost:.3f}"
)

y_pred = (
    y_proba >= best_threshold_catboost
).astype(int)

metrics_catboost = evaluate_model(
    y_test,
    y_pred,
    y_proba,
    "CatBoost"
)

feature_importance(importance_pred_val, FEATURES, False)

plot_roc_curve(y_test, y_proba, 'Catboost', metrics_catboost["ROC-AUC"])

plot_feature_importance(importance_pred_val, FEATURES)

plot_threshold_metrics(
    threshold_results_catboost
)

plot_confusion_matrix(
    y_test,
    y_pred,
    "CatBoost"
)

save_model_metrics(
    "CatBoost",
    metrics_catboost,
    best_threshold_catboost
)

save_confusion_matrix(
    y_test,
    y_pred,
    "CatBoost",
    threshold=best_threshold_catboost,
)

Path("../../models").mkdir(
    parents=True,
    exist_ok=True,
)

model.save_model(
    "../../models/catboost_model.cbm"
)