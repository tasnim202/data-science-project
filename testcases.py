"""
CSE352 – Data Science Final Project
Automated Test Cases

Supports BOTH Classification and Regression tasks.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


def test_pipeline_used(model):
    assert isinstance(model, Pipeline), (
        "Model must be wrapped inside a scikit-learn Pipeline."
    )


def test_column_transformer_used(model):
    preprocessor = model.named_steps.get("preprocessor", None)
    assert isinstance(preprocessor, ColumnTransformer), (
        "Preprocessing must be done using ColumnTransformer."
    )


def test_categorical_encoding(model):
    preprocessor = model.named_steps["preprocessor"]
    transformers = [t[1] for t in preprocessor.transformers]

    encoder_found = any(
        isinstance(t, OneHotEncoder) or
        (hasattr(t, "steps") and any(isinstance(s[1], OneHotEncoder) for s in t.steps))
        for t in transformers
    )

    assert encoder_found, (
        "Categorical features must be encoded using OneHotEncoder."
    )


def test_feature_scaling(model):
    preprocessor = model.named_steps["preprocessor"]
    transformers = [t[1] for t in preprocessor.transformers]

    scaler_found = any(
        isinstance(t, (StandardScaler, MinMaxScaler)) or
        (hasattr(t, "steps") and any(isinstance(s[1], (StandardScaler, MinMaxScaler)) for s in t.steps))
        for t in transformers
    )

    assert scaler_found, (
        "Numerical features must be scaled using StandardScaler or MinMaxScaler."
    )


def test_no_data_leakage(model):
    steps = list(model.named_steps.keys())
    assert steps[0] == "preprocessor", (
        "Preprocessing must be the FIRST step in the pipeline."
    )
    assert "model" in steps[-1], (
        "Final step in the pipeline must be the model."
    )


def test_train_test_split(X_train, X_test, y_train, y_test):
    assert len(X_train) > 0 and len(X_test) > 0, (
        "Train/test split must be performed."
    )
    assert len(X_train) != len(X_test), (
        "Training and testing sets must be different."
    )


def test_classification_metrics(metrics_dict):
    required_metrics = {
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC"
    }

    assert isinstance(metrics_dict, dict), (
        "Evaluation metrics must be returned as a dictionary."
    )

    missing = required_metrics - metrics_dict.keys()
    assert not missing, f"Missing classification metrics: {missing}"


def test_regression_metrics(metrics_dict):
    required_metrics = {
        "MSE",
        "RMSE",
        "MAE"
    }

    assert isinstance(metrics_dict, dict), (
        "Evaluation metrics must be returned as a dictionary."
    )

    missing = required_metrics - metrics_dict.keys()
    assert not missing, f"Missing regression metrics: {missing}"
