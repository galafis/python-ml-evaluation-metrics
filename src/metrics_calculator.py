"""ML Model Evaluation Metrics Calculator.

Comprehensive toolkit for classification and regression metrics.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    log_loss,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate evaluation metrics for ML models."""

    def evaluate_classification(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        y_prob: Optional[Union[List, np.ndarray]] = None,
        average: str = "binary",
    ) -> Dict:
        """Calculate classification metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_prob: Predicted probabilities (optional).
            average: Averaging method for multiclass.

        Returns:
            Dictionary with all classification metrics.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        results = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        }

        if y_prob is not None:
            y_prob = np.array(y_prob)
            try:
                results["roc_auc"] = float(roc_auc_score(y_true, y_prob))
                fpr, tpr, thresholds = roc_curve(y_true, y_prob)
                results["fpr"] = fpr.tolist()
                results["tpr"] = tpr.tolist()
                results["roc_thresholds"] = thresholds.tolist()
                precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_true, y_prob)
                results["pr_precision"] = precision_vals.tolist()
                results["pr_recall"] = recall_vals.tolist()
                results["pr_auc"] = float(average_precision_score(y_true, y_prob))
                results["log_loss"] = float(log_loss(y_true, y_prob))
            except ValueError as e:
                logger.warning(f"Could not compute probability metrics: {e}")

        logger.info(f"Classification metrics calculated: accuracy={results['accuracy']:.4f}")
        return results

    def evaluate_regression(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        n_features: Optional[int] = None,
    ) -> Dict:
        """Calculate regression metrics.

        Args:
            y_true: True values.
            y_pred: Predicted values.
            n_features: Number of features (for adjusted R2).

        Returns:
            Dictionary with all regression metrics.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        mse = float(mean_squared_error(y_true, y_pred))
        results = {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2_score": float(r2_score(y_true, y_pred)),
            "mape": float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-10, None))) * 100),
        }

        if n_features is not None:
            n = len(y_true)
            r2 = results["r2_score"]
            results["adjusted_r2"] = float(1 - (1 - r2) * (n - 1) / (n - n_features - 1))

        logger.info(f"Regression metrics calculated: RMSE={results['rmse']:.4f}")
        return results

    def cross_validate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = "accuracy",
    ) -> Dict:
        """Perform cross-validation.

        Args:
            model: Sklearn-compatible model.
            X: Feature matrix.
            y: Target vector.
            cv: Number of folds.
            scoring: Scoring metric.

        Returns:
            Cross-validation results with confidence intervals.
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)

        results = {
            "scores": scores.tolist(),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "ci_95_lower": float(np.mean(scores) - 1.96 * np.std(scores) / np.sqrt(cv)),
            "ci_95_upper": float(np.mean(scores) + 1.96 * np.std(scores) / np.sqrt(cv)),
            "n_folds": cv,
            "scoring": scoring,
        }

        logger.info(f"Cross-validation: {scoring}={results['mean']:.4f} (+/- {results['std']:.4f})")
        return results


def main():
    """Demonstrate metrics calculator usage."""
    calculator = MetricsCalculator()

    # Binary classification example
    y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1, 1, 1, 1, 0, 0]
    y_prob = [0.1, 0.9, 0.4, 0.2, 0.85, 0.6, 0.75, 0.95, 0.15, 0.3]

    print("=" * 60)
    print("ML Model Evaluation Metrics Calculator")
    print("=" * 60)

    clf_results = calculator.evaluate_classification(y_true, y_pred, y_prob)
    print(f"\nClassification Results:")
    print(f"  Accuracy:  {clf_results['accuracy']:.4f}")
    print(f"  Precision: {clf_results['precision']:.4f}")
    print(f"  Recall:    {clf_results['recall']:.4f}")
    print(f"  F1-Score:  {clf_results['f1_score']:.4f}")
    print(f"  ROC-AUC:   {clf_results.get('roc_auc', 'N/A')}")
    print(f"  Log Loss:  {clf_results.get('log_loss', 'N/A')}")
    print(f"\nConfusion Matrix:")
    for row in clf_results['confusion_matrix']:
        print(f"  {row}")

    # Regression example
    y_true_reg = [3.0, 5.0, 2.5, 7.0, 4.5]
    y_pred_reg = [2.8, 5.2, 2.1, 6.8, 4.9]

    reg_results = calculator.evaluate_regression(y_true_reg, y_pred_reg, n_features=2)
    print(f"\nRegression Results:")
    print(f"  MSE:          {reg_results['mse']:.4f}")
    print(f"  RMSE:         {reg_results['rmse']:.4f}")
    print(f"  MAE:          {reg_results['mae']:.4f}")
    print(f"  R2 Score:     {reg_results['r2_score']:.4f}")
    print(f"  Adjusted R2:  {reg_results.get('adjusted_r2', 'N/A')}")


if __name__ == "__main__":
    main()
