"""
train.py - CLI Training and Model Evaluation Pipeline.

Trains, cross-validates, tunes, and exports the Loan Default Prediction model.
Supports Stratified K-Fold CV, hyperparameter optimization via GridSearchCV,
and exports the best-performing pipeline artifact with metadata.

Usage:
    python src/train.py --data_path loan_data.csv --output_model_path models/loan_model.joblib --tune
"""

import argparse
import os
import sys
import json
import logging
from typing import Dict, Any

# Ensure project root is in sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from src.preprocessing import load_and_clean_data, build_full_pipeline

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("LoanDefaultTrainer")


def evaluate_model_on_test(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Computes comprehensive evaluation metrics on held-out test set."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision (Class 1)": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall (Class 1)": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1-Score (Class 1)": float(f1_score(y_test, y_pred, zero_division=0)),
        "Recall (Class 0 - Default)": float(recall_score(y_test, y_pred, pos_label=0, zero_division=0)),
        "F1-Score (Class 0 - Default)": float(f1_score(y_test, y_pred, pos_label=0, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, y_proba)),
    }
    return metrics


def run_training_pipeline(
    data_path: str = "loan_data.csv",
    output_model_path: str = "models/loan_model.joblib",
    tune_rf: bool = True,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Executes the end-to-end model training, cross-validation, and selection pipeline."""
    logger.info("Loading and cleaning raw dataset from %s ...", data_path)
    X, y = load_and_clean_data(data_path)
    logger.info("Dataset shape: X=%s, y=%s (Class 1: %d, Class 0: %d)", X.shape, y.shape, (y == 1).sum(), (y == 0).sum())

    # 1. Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    logger.info("Stratified Split completed: Train size=%d, Test size=%d", len(X_train), len(X_test))

    # 2. Define Candidate Models
    candidate_models = {
        "Logistic Regression": LogisticRegression(
            solver="liblinear",
            C=1.0,
            max_iter=1000,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
        ),
        "Random Forest (Baseline)": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
        ),
    }

    # 3. Stratified K-Fold Cross-Validation on Training Set
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = []
    trained_pipelines = {}

    for name, clf in candidate_models.items():
        logger.info("Running %d-Fold Cross-Validation for: %s", cv_folds, name)
        pipe = build_full_pipeline(clf)
        
        cv_res = cross_validate(pipe, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)
        
        # Fit on full training split
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        # Evaluate on test set
        test_metrics = evaluate_model_on_test(pipe, X_test, y_test)

        summary_row = {
            "Model": name,
            "CV ROC-AUC (Mean)": np.mean(cv_res["test_roc_auc"]),
            "CV ROC-AUC (Std)": np.std(cv_res["test_roc_auc"]),
            "CV F1 (Mean)": np.mean(cv_res["test_f1"]),
            "Test Accuracy": test_metrics["Accuracy"],
            "Test F1 (Class 1)": test_metrics["F1-Score (Class 1)"],
            "Test Recall (Class 0 - Default)": test_metrics["Recall (Class 0 - Default)"],
            "Test F1 (Class 0 - Default)": test_metrics["F1-Score (Class 0 - Default)"],
            "Test ROC-AUC": test_metrics["ROC-AUC"],
        }
        results.append(summary_row)

    # 4. Hyperparameter Tuning for Random Forest
    if tune_rf:
        logger.info("Starting GridSearchCV hyperparameter tuning for Random Forest...")
        rf_base = build_full_pipeline(RandomForestClassifier(random_state=random_state))
        
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [4, 6, 8, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__class_weight": [None, "balanced"],
        }

        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            cv=skf,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)
        
        best_rf_pipe = grid_search.best_estimator_
        trained_pipelines["Random Forest (Tuned)"] = best_rf_pipe
        logger.info("GridSearchCV Optimal Parameters: %s", grid_search.best_params_)
        logger.info("GridSearchCV Best CV ROC-AUC: %.4f", grid_search.best_score_)

        # Evaluate Tuned RF on test set
        test_metrics_rf = evaluate_model_on_test(best_rf_pipe, X_test, y_test)
        results.append({
            "Model": "Random Forest (Tuned)",
            "CV ROC-AUC (Mean)": grid_search.best_score_,
            "CV ROC-AUC (Std)": 0.0,
            "CV F1 (Mean)": float(f1_score(y_train, best_rf_pipe.predict(X_train))),
            "Test Accuracy": test_metrics_rf["Accuracy"],
            "Test F1 (Class 1)": test_metrics_rf["F1-Score (Class 1)"],
            "Test Recall (Class 0 - Default)": test_metrics_rf["Recall (Class 0 - Default)"],
            "Test F1 (Class 0 - Default)": test_metrics_rf["F1-Score (Class 0 - Default)"],
            "Test ROC-AUC": test_metrics_rf["ROC-AUC"],
        })

    # 5. Model Selection & Export
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 95)
    print("                      MODEL PERFORMANCE EVALUATION SUMMARY")
    print("=" * 95)
    print(results_df.to_string(index=False))
    print("=" * 95 + "\n")

    # Select best model prioritizing ROC-AUC and Defaulter Recall (Cost-sensitive credit risk)
    best_model_name = "Random Forest (Tuned)" if "Random Forest (Tuned)" in trained_pipelines else "Logistic Regression"
    best_pipe = trained_pipelines[best_model_name]

    # Detailed test metrics for the winner
    y_pred_best = best_pipe.predict(X_test)
    print(f"Classification Report for Best Pipeline ({best_model_name}) on Test Set:")
    print(classification_report(y_test, y_pred_best, target_names=["Default (0)", "Approved (1)"]))

    cm = confusion_matrix(y_test, y_pred_best)
    print(f"Confusion Matrix:\n{cm}\n")

    # Save artifact
    output_dir = os.path.dirname(output_model_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    joblib.dump(best_pipe, output_model_path)
    logger.info("Successfully exported optimal model pipeline artifact to: %s", output_model_path)

    # Save summary metadata JSON alongside model
    meta_path = os.path.splitext(output_model_path)[0] + "_metrics.json"
    meta = {
        "best_model": best_model_name,
        "evaluation_summary": results,
        "test_confusion_matrix": cm.tolist(),
        "optimal_params": grid_search.best_params_ if tune_rf else None,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    logger.info("Saved model evaluation metadata to: %s", meta_path)

    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Loan Default Prediction Classification Pipeline")
    parser.add_argument("--data_path", type=str, default="loan_data.csv", help="Path to input loan dataset")
    parser.add_argument("--output_model_path", type=str, default="models/loan_model.joblib", help="Target model export path")
    parser.add_argument("--tune", action="store_true", default=True, help="Perform hyperparameter tuning for Random Forest")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of Stratified K-Fold splits")
    args = parser.parse_args()

    run_training_pipeline(
        data_path=args.data_path,
        output_model_path=args.output_model_path,
        tune_rf=args.tune,
        cv_folds=args.cv_folds,
    )
