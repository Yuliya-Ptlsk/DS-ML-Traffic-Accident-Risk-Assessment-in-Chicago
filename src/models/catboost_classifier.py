from catboost import  CatBoostClassifier, Pool


def train_catboost(X_train, y_train, X_test, y_test, cat_features):
    model = CatBoostClassifier(
        iterations=1000,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        auto_class_weights="Balanced",
        verbose=100,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
    )

    y_pred = model.predict(
        X_test
    )

    y_proba = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    importance_pred_val = model.get_feature_importance(
        type="PredictionValuesChange"
    )

    train_pool = Pool(
        X_train,
        y_train,
        cat_features=cat_features
    )

    importance_loss = (
        model.get_feature_importance(
            train_pool,
            type="LossFunctionChange"
        )
    )

    test_pool = Pool(
        X_test,
        y_test,
        cat_features=cat_features
    )

    shap_values = (
        model.get_feature_importance(
            test_pool,
            type="ShapValues"
        )
    )

    return (
        model,
        y_proba,
        y_pred,
        importance_pred_val,
        importance_loss,
        shap_values
    )
