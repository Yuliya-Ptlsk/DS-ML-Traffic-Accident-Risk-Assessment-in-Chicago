import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def evaluate_model(
    y_true,
    y_pred,
    y_proba,
    model_name,
):
    metrics = {
        "ROC-AUC": roc_auc_score(
            y_true,
            y_proba,
        ),
        "PR_AUC": average_precision_score(
            y_true,
            y_proba,
        ),
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }

    print(f"\n {model_name} Classification Report")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    print(f"\n {model_name} Metrics")

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics


def feature_importance(importances, features, abs_flag=True):
    features = pd.DataFrame(
        {
            "feature": features,
            "importance": importances
        }
    )

    if abs_flag:
        features["abs_importance"] = (
            features["importance"].abs()
        )

        features = (
            features
            .sort_values(
                "abs_importance",
                ascending=False
            )
        )
    else:
        features = (
            features
            .sort_values(
                "importance",
                ascending=False
            )
        )


    print(
        features.head(15)
    )

def save_model_metrics(
    model_name,
    metrics,
    threshold=None,
    file_path="../../data/interim/model_metrics.csv",
):
    row = {
        "model": model_name,
        "threshold": threshold,
        "accuracy": metrics["Accuracy"],
        "precision": metrics["Precision"],
        "recall": metrics["Recall"],
        "f1": metrics["F1"],
        "roc_auc": metrics["ROC-AUC"],
        "pr_auc": metrics["PR_AUC"],
    }

    df_row = pd.DataFrame([row])

    try:
        existing = pd.read_csv(
            file_path
        )

        # update existing record
        existing = existing[
            existing["model"] != model_name
        ]

        df = pd.concat(
            [existing, df_row],
            ignore_index=True,
        )

    except FileNotFoundError:
        df = df_row

    df.to_csv(
        file_path,
        index=False,
    )
