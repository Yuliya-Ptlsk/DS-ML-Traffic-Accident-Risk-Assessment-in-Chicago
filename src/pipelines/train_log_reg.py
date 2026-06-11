import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.log_reg import train_log_reg
from src.models.metrics import evaluate_model, feature_importance, save_model_metrics
from src.utils.conf_matrix import save_confusion_matrix
from src.visualization.plots import (
    plot_roc_curve,
    plot_pr_curve,
    plot_feature_importance,
    plot_confusion_matrix
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

# ----------------------------------------------------
# TRAIN LOGISTIC REGRESSION MODEL ON NUMERIC FEATURES
# ----------------------------------------------------
X = df[FEATURES_NUMERIC]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

log_reg_1 = train_log_reg(
    X_train,
    X_test,
    y_train,
    FEATURES_NUMERIC,
)

y_proba_log_reg_1, y_pred_log_reg_1, coefficients_log_reg_1, feature_names_1 = log_reg_1

metrics_log_reg_1 = evaluate_model(
    y_test,
    y_pred_log_reg_1,
    y_proba_log_reg_1,
    "Logistic Regression Numeric",
)

plot_roc_curve(y_test, y_proba_log_reg_1, 'Logistic Regression Numeric', metrics_log_reg_1["ROC-AUC"])
# plot_pr_curve(y_test, y_proba_log_reg, 'Logistic Regression', metrics_log_reg["PR_AUC"])

feature_importance(coefficients_log_reg_1, feature_names_1)

plot_feature_importance(coefficients_log_reg_1, feature_names_1)

plot_confusion_matrix(
    y_test,
    y_pred_log_reg_1,
    "Logistic Regression Numeric"
)

save_model_metrics(
    "LogReg Numeric",
    metrics_log_reg_1
)

save_confusion_matrix(
    y_test,
    y_pred_log_reg_1,
    "LogReg Numeric",
)


# ----------------------------------------------------------------------------
# TRAIN LOGISTIC REGRESSION MODEL ON NUMERIC FEATURES + CATEGORICAL FEATURES
# ----------------------------------------------------------------------------
X = df[FEATURES]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

log_reg_2 = train_log_reg(
    X_train,
    X_test,
    y_train,
    FEATURES_NUMERIC,
    FEATURES_CATEGORICAL,
)

y_proba_log_reg_2, y_pred_log_reg_2, coefficients_log_reg_2, feature_names_2 = log_reg_2

metrics_log_reg_2 = evaluate_model(
    y_test,
    y_pred_log_reg_2,
    y_proba_log_reg_2,
    "Logistic Regression Numeric + Categorical",
)

plot_roc_curve(y_test, y_proba_log_reg_2, 'Logistic Regression Numeric + Categorical', metrics_log_reg_2["ROC-AUC"])
# plot_pr_curve(y_test, y_proba_log_reg, 'Logistic Regression', metrics_log_reg["PR_AUC"])

feature_importance(coefficients_log_reg_2, feature_names_2)

plot_feature_importance(coefficients_log_reg_2, feature_names_2)

plot_confusion_matrix(
    y_test,
    y_pred_log_reg_2,
    "Logistic Regression Numeric + Segment"
)

save_model_metrics(
    "LogReg Numeric + Segment",
    metrics_log_reg_2
)

save_confusion_matrix(
    y_test,
    y_pred_log_reg_2,
    "LogReg Numeric + Segment",
)