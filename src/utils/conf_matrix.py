import pandas as pd
from sklearn.metrics import confusion_matrix
from pathlib import Path

def get_confusion_matrix(
    y_true,
    y_pred,
):
    return confusion_matrix(
        y_true,
        y_pred
    )

def save_confusion_matrix(
    y_true,
    y_pred,
    model_name,
    threshold=None,
    file_path="../../data/interim/confusion_matrix.csv",
):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    result = pd.DataFrame(
        [
            {
                "model": model_name,
                "threshold": threshold,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        ]
    )

    path = Path(file_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        existing = pd.read_csv(path)

        existing = existing[
            existing["model"] != model_name
        ]

        result = pd.concat(
            [existing, result],
            ignore_index=True,
        )

    result.to_csv(
        path,
        index=False,
    )