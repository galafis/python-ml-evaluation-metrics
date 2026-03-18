"""Tests for MetricsCalculator."""

import pytest
import numpy as np

from src.metrics_calculator import MetricsCalculator


class TestClassificationMetrics:
    """Test classification metric calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = MetricsCalculator()
        self.y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
        self.y_pred = [0, 1, 0, 0, 1, 1, 1, 1, 0, 0]
        self.y_prob = [0.1, 0.9, 0.4, 0.2, 0.85, 0.6, 0.75, 0.95, 0.15, 0.3]

    def test_basic_classification_metrics(self):
        """Test basic classification metrics are computed."""
        results = self.calculator.evaluate_classification(
            self.y_true, self.y_pred
        )
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1_score" in results
        assert "confusion_matrix" in results
        assert 0 <= results["accuracy"] <= 1
        assert 0 <= results["precision"] <= 1
        assert 0 <= results["recall"] <= 1
        assert 0 <= results["f1_score"] <= 1

    def test_classification_with_probabilities(self):
        """Test classification with probability scores."""
        results = self.calculator.evaluate_classification(
            self.y_true, self.y_pred, self.y_prob
        )
        assert "roc_auc" in results
        assert "fpr" in results
        assert "tpr" in results
        assert "pr_auc" in results
        assert "log_loss" in results
        assert 0 <= results["roc_auc"] <= 1

    def test_perfect_classification(self):
        """Test metrics for perfect predictions."""
        y = [0, 1, 0, 1]
        results = self.calculator.evaluate_classification(y, y)
        assert results["accuracy"] == 1.0
        assert results["f1_score"] == 1.0

    def test_confusion_matrix_shape(self):
        """Test confusion matrix has correct shape."""
        results = self.calculator.evaluate_classification(
            self.y_true, self.y_pred
        )
        cm = results["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2


class TestRegressionMetrics:
    """Test regression metric calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = MetricsCalculator()
        self.y_true = [3.0, 5.0, 2.5, 7.0, 4.5]
        self.y_pred = [2.8, 5.2, 2.1, 6.8, 4.9]

    def test_basic_regression_metrics(self):
        """Test basic regression metrics."""
        results = self.calculator.evaluate_regression(
            self.y_true, self.y_pred
        )
        assert "mse" in results
        assert "rmse" in results
        assert "mae" in results
        assert "r2_score" in results
        assert results["mse"] >= 0
        assert results["rmse"] >= 0
        assert results["mae"] >= 0

    def test_rmse_is_sqrt_mse(self):
        """Test RMSE is square root of MSE."""
        results = self.calculator.evaluate_regression(
            self.y_true, self.y_pred
        )
        assert abs(results["rmse"] - np.sqrt(results["mse"])) < 1e-10

    def test_adjusted_r2(self):
        """Test adjusted R2 with n_features."""
        results = self.calculator.evaluate_regression(
            self.y_true, self.y_pred, n_features=2
        )
        assert "adjusted_r2" in results

    def test_perfect_regression(self):
        """Test perfect regression predictions."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        results = self.calculator.evaluate_regression(y, y)
        assert results["mse"] == 0.0
        assert results["r2_score"] == 1.0


class TestVisualization:
    """Test visualization module."""

    def test_import_visualizer(self):
        """Test that visualizer can be imported."""
        from src.visualization import MetricsVisualizer
        viz = MetricsVisualizer()
        assert viz is not None

    def test_confusion_matrix_plot(self):
        """Test confusion matrix plotting."""
        from src.visualization import MetricsVisualizer
        viz = MetricsVisualizer()
        cm = [[3, 1], [1, 5]]
        fig = viz.plot_confusion_matrix(cm)
        assert fig is not None

    def test_roc_curve_plot(self):
        """Test ROC curve plotting."""
        from src.visualization import MetricsVisualizer
        viz = MetricsVisualizer()
        fig = viz.plot_roc_curve(
            fpr=[0.0, 0.2, 0.5, 1.0],
            tpr=[0.0, 0.6, 0.8, 1.0],
            auc_score=0.85
        )
        assert fig is not None
