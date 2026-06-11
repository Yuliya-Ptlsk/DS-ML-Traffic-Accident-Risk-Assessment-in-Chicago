import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from src.models.mlp import train_mlp
from src.models.metrics import evaluate_model, save_model_metrics
from src.utils.conf_matrix import save_confusion_matrix
from src.visualization.plots import (
    plot_roc_curve,
    plot_mlp_loss_curve,
    plot_mlp_validation_curve,
    plot_threshold_metrics, plot_confusion_matrix,
)
from src.models.threshold import (
    find_best_threshold
)


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
    "segment_id"
]

FEATURES = FEATURES_NUMERIC + FEATURES_CATEGORICAL

TARGET = "accident"

y = df[TARGET]

ros = RandomOverSampler(
    random_state=42
)

# ---------------------------------------------
# TRAIN MLP ON NUMERIC FEATURES
# ---------------------------------------------
X = df[FEATURES_NUMERIC]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

X_train_balanced, y_train_balanced = ros.fit_resample(
    X_train,
    y_train
)

mlp_1 = train_mlp(
    X_train_balanced,
    y_train_balanced,
    X_test,
    FEATURES_NUMERIC
)

y_proba_mlp_1, y_pred_mlp_1, pipeline_mlp_1 = mlp_1

best_threshold_mlp_1, threshold_results_mlp_1 = (
    find_best_threshold(
        y_test,
        y_proba_mlp_1
    )
)

print(
    f"Best threshold: {best_threshold_mlp_1:.3f}"
)

plot_threshold_metrics(
    threshold_results_mlp_1
)

y_pred_best_mlp_1 = (
    y_proba_mlp_1 >= best_threshold_mlp_1
).astype(int)

metrics_mlp_1 = evaluate_model(
    y_test,
    y_pred_best_mlp_1,
    y_proba_mlp_1,
    "MLP Numeric"
)

plot_roc_curve(
    y_test,
    y_proba_mlp_1,
    "MLP Numeric",
    metrics_mlp_1["ROC-AUC"]
)

plot_mlp_loss_curve(
    pipeline_mlp_1,
    "MLP Numeric Training Loss",
)

plot_mlp_validation_curve(
    pipeline_mlp_1,
    "MLP Numeric Validation Score"
)

plot_confusion_matrix(
    y_test,
    y_pred_best_mlp_1,
    "MLP Numeric"
)

save_model_metrics(
    "MLP Numeric",
    metrics_mlp_1,
    best_threshold_mlp_1
)

save_confusion_matrix(
    y_test,
    y_pred_best_mlp_1,
    "MLP Numeric",
    threshold=best_threshold_mlp_1,
)


# -----------------------------------------------------
# TRAIN MLP ON NUMERIC FEATURES + CATEGORICAL FEATURES
# -----------------------------------------------------
X = df[FEATURES]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

X_train_balanced, y_train_balanced = ros.fit_resample(
    X_train,
    y_train
)

mlp_2 = train_mlp(
    X_train_balanced,
    y_train_balanced,
    X_test,
    FEATURES_NUMERIC,
    FEATURES_CATEGORICAL,
)

y_proba_mlp_2, y_pred_mlp_2, pipeline_mlp_2 = mlp_2

best_threshold_mlp_2, threshold_results_mlp_2 = (
    find_best_threshold(
        y_test,
        y_proba_mlp_2
    )
)

print(
    f"Best threshold: {best_threshold_mlp_2:.3f}"
)

plot_threshold_metrics(
    threshold_results_mlp_2
)

y_pred_best_mlp_2 = (
    y_proba_mlp_2 >= best_threshold_mlp_2
).astype(int)

metrics_mlp_2 = evaluate_model(
    y_test,
    y_pred_best_mlp_2,
    y_proba_mlp_2,
    "MLP Numeric + Categorical"
)

plot_roc_curve(
    y_test,
    y_proba_mlp_2,
    "MLP Numeric + Categorical",
    metrics_mlp_2["ROC-AUC"]
)

plot_mlp_loss_curve(
    pipeline_mlp_2,
"MLP Numeric + Categorical Training Loss"
)

plot_mlp_validation_curve(
    pipeline_mlp_2,
    "MLP Numeric + Categorical Validation Score",
)

plot_confusion_matrix(
    y_test,
    y_pred_best_mlp_2,
    "MLP Numeric + Segment"
)

save_model_metrics(
    "MLP Numeric + Segment",
    metrics_mlp_2,
    best_threshold_mlp_2
)

save_confusion_matrix(
    y_test,
    y_pred_best_mlp_2,
    "MLP Numeric + Segment",
    threshold=best_threshold_mlp_2,
)