"""Unit tests for evaluation metrics."""

import numpy as np
import pytest
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from evaluation.metrics import BaseMetrics, RegressionMetrics, ClassificationMetrics


class TestBaseMetrics:
    """Test cases for the BaseMetrics class."""
    
    def test_base_metrics_evaluate_not_implemented(self):
        """Test that BaseMetrics.evaluate raises NotImplementedError."""
        metrics = BaseMetrics()
        with pytest.raises(NotImplementedError):
            metrics.evaluate(np.array([1, 2]), np.array([1, 2]))


class TestRegressionMetrics:
    """Test cases for the RegressionMetrics class."""
    
    def test_initialization(self):
        """Test initialization of RegressionMetrics."""
        metrics = RegressionMetrics()
        assert isinstance(metrics, BaseMetrics)
        
    def test_evaluate(self):
        """Test evaluate method with simple data."""
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        
        metrics = RegressionMetrics()
        results = metrics.evaluate(y_true, y_pred)
        
        # Calculate expected values
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        explained_variance = 1 - (np.var(y_true - y_pred) / np.var(y_true))
        
        # Check results
        assert "mae" in results
        assert "mse" in results
        assert "rmse" in results
        assert "r2" in results
        assert "explained_variance" in results
        
        # Compare with expected values
        assert results["mae"] == pytest.approx(mae)
        assert results["mse"] == pytest.approx(mse)
        assert results["rmse"] == pytest.approx(rmse)
        assert results["r2"] == pytest.approx(r2)
        assert results["explained_variance"] == pytest.approx(explained_variance)
        
    def test_evaluate_with_lists(self):
        """Test evaluate method with list inputs."""
        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        
        metrics = RegressionMetrics()
        results = metrics.evaluate(y_true, y_pred)
        
        # Calculate expected values
        mae = mean_absolute_error(y_true, y_pred)
        
        # Check a sample metric
        assert results["mae"] == pytest.approx(mae)
        
    def test_evaluate_perfect_prediction(self):
        """Test evaluate method with perfect predictions."""
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        
        metrics = RegressionMetrics()
        results = metrics.evaluate(y_true, y_pred)
        
        # Perfect prediction should have these values
        assert results["mae"] == 0
        assert results["mse"] == 0
        assert results["rmse"] == 0
        assert results["r2"] == 1
        assert results["explained_variance"] == 1


class TestClassificationMetrics:
    """Test cases for the ClassificationMetrics class."""
    
    def test_initialization_default(self):
        """Test initialization of ClassificationMetrics with default parameters."""
        metrics = ClassificationMetrics()
        assert isinstance(metrics, BaseMetrics)
        assert metrics.average == "binary"
        
    def test_initialization_custom_average(self):
        """Test initialization of ClassificationMetrics with custom average."""
        metrics = ClassificationMetrics(average="macro")
        assert metrics.average == "macro"
        
    def test_evaluate_binary(self):
        """Test evaluate method with binary classification data."""
        y_true = np.array([0, 1, 0, 1, 0, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 1])
        
        metrics = ClassificationMetrics()
        results = metrics.evaluate(y_true, y_pred)
        
        # Calculate expected values
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
        recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()
        
        # Check results
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        assert "confusion_matrix" in results
        
        # Compare with expected values
        assert results["accuracy"] == pytest.approx(accuracy)
        assert results["precision"] == pytest.approx(precision)
        assert results["recall"] == pytest.approx(recall)
        assert results["f1"] == pytest.approx(f1)
        assert results["confusion_matrix"] == cm
        
    def test_evaluate_with_probabilities_binary(self):
        """Test evaluate method with probabilities for binary classification."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        y_prob = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.9, 0.1],
            [0.6, 0.4]
        ])
        
        metrics = ClassificationMetrics()
        results = metrics.evaluate(y_true, y_pred, y_prob)
        
        # Calculate expected ROC AUC
        expected_roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        
        # Check ROC AUC is included and correct
        assert "roc_auc" in results
        assert results["roc_auc"] == pytest.approx(expected_roc_auc)
        
    def test_evaluate_multiclass(self):
        """Test evaluate method with multiclass classification data."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 1, 2])
        
        metrics = ClassificationMetrics(average="macro")
        results = metrics.evaluate(y_true, y_pred)
        
        # Calculate expected values
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        
        # Compare with expected values
        assert results["accuracy"] == pytest.approx(accuracy)
        assert results["precision"] == pytest.approx(precision)
        assert results["recall"] == pytest.approx(recall)
        assert results["f1"] == pytest.approx(f1)
        
    def test_evaluate_with_probabilities_multiclass(self):
        """Test evaluate method with probabilities for multiclass classification."""
        y_true = np.array([0, 1, 2, 0])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.6, 0.2],
            [0.7, 0.2, 0.1]
        ])
        
        metrics = ClassificationMetrics(average="macro")
        results = metrics.evaluate(y_true, y_pred, y_prob)
        
        # Calculate expected ROC AUC (this is complex for multiclass)
        try:
            expected_roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average="macro")
            # Check ROC AUC is included and correct
            assert "roc_auc" in results
            assert results["roc_auc"] == pytest.approx(expected_roc_auc)
        except (ValueError, IndexError):
            # If sklearn can't calculate it, our metrics should also skip it
            assert "roc_auc" not in results
    
    def test_evaluate_with_invalid_probabilities(self):
        """Test evaluate method with invalid probabilities."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        # Invalid probabilities (wrong shape)
        y_prob = np.array([0.2, 0.7, 0.1, 0.4])
        
        metrics = ClassificationMetrics()
        results = metrics.evaluate(y_true, y_pred, y_prob)
        
        # Should compute normal metrics but skip ROC AUC
        assert "accuracy" in results
        assert "roc_auc" not in results  # ROC AUC should be skipped
