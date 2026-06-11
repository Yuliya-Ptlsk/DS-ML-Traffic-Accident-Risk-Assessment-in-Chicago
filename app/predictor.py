from pathlib import Path

from catboost import CatBoostClassifier

def load_model():

    model = CatBoostClassifier()

    model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "catboost_model.cbm"
    )

    model.load_model(model_path)

    return model