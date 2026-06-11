import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


def find_best_threshold(
    y_true,
    y_proba,
    start=0.01,
    stop=0.99,
    step=0.01,
):
    results = []

    thresholds = np.arange(
        start,
        stop,
        step
    )

    for threshold in thresholds:

        y_pred = (
            y_proba >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    results = pd.DataFrame(results)

    best_row = (
        results
        .sort_values(
            "f1",
            ascending=False
        )
        .iloc[0]
    )

    print("\nBest threshold")
    print(best_row)

    return (
        float(best_row["threshold"]),
        results,
    )