"""Integration tests for the ML library pipeline."""

import numpy as np
import pytest

from ml_library.models import (
    LinearRegression, LogisticRegression,
    SVMRegressor, SVMClassifier,
    KNNRegressor, KNNClassifier
)
from ml_library.training import Trainer
from ml_library.evaluation import RegressionMetrics, ClassificationMetrics
from ml_library.inference import Predictor


def test_regression_pipeline(sample_regression_data):
    """Test the full regression pipeline from training to evaluation."""
    X, y, _ = sample_regression_data
    
    # Split data
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    test_size = int(0.2 * n_samples)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # Create model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    
    # Train the model
    trainer = Trainer(model, validation_split=0.0)
    trained_model = trainer.train(X_train, y_train)
    
    # Get metrics
    metrics = trainer.get_metrics()
    assert "train_score" in metrics
    
    # Create predictor
    predictor = Predictor(trained_model)
    y_pred = predictor.predict(X_test)
    
    # Evaluate predictions
    evaluator = RegressionMetrics()
    results = evaluator.evaluate(y_test, y_pred)
    
    # Check that all expected metrics are present
    assert "mae" in results
    assert "mse" in results
    assert "rmse" in results
    assert "r2" in results
    
    # Check that r2 score is reasonable
    assert results["r2"] > 0.5  # Should be a decent fit for linear data


def test_classification_pipeline(sample_classification_data):
    """Test the full classification pipeline from training to evaluation."""
    X, y = sample_classification_data
    
    # Split data
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    test_size = int(0.2 * n_samples)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # Create model
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    
    # Train the model
    trainer = Trainer(model, validation_split=0.0)
    trained_model = trainer.train(X_train, y_train)
    
    # Get metrics
    metrics = trainer.get_metrics()
    assert "train_score" in metrics
    
    # Create predictor
    predictor = Predictor(trained_model)
    result = predictor.predict(X_test, return_probabilities=True)
    assert isinstance(result, dict)
    assert "predictions" in result
    assert "probabilities" in result
    
    y_pred = result["predictions"]
    y_prob = result["probabilities"]
    
    # Evaluate predictions
    evaluator = ClassificationMetrics()
    results = evaluator.evaluate(y_test, y_pred, y_prob)
    
    # Check that all expected metrics are present
    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1" in results
    assert "confusion_matrix" in results
    
    # Check that accuracy is reasonable
    assert results["accuracy"] > 0.7  # Should be decent for linearly separable data


@pytest.mark.parametrize(
    "model",
    [
        SVMRegressor(kernel="linear", C=1.0),
        KNNRegressor(n_neighbors=3)
    ]
)
def test_advanced_regression_pipeline(sample_regression_data, model):
    """Test regression pipeline with SVM and KNN regression models."""
    X, y, _ = sample_regression_data
    
    # Split data
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    test_size = int(0.2 * n_samples)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # Train the model
    trainer = Trainer(model, validation_split=0.2)
    trained_model = trainer.train(X_train, y_train)
    
    # Get metrics
    metrics = trainer.get_metrics()
    assert "train_score" in metrics
    assert "val_score" in metrics
    
    # Create predictor
    predictor = Predictor(trained_model)
    y_pred = predictor.predict(X_test)
    
    # Evaluate predictions
    evaluator = RegressionMetrics()
    results = evaluator.evaluate(y_test, y_pred)
    
    # Check that all expected metrics are present
    assert "mae" in results
    assert "mse" in results
    assert "rmse" in results
    assert "r2" in results
    
    # Check that r2 score is reasonable
    assert results["r2"] > 0  # Should be at least positive correlation


@pytest.mark.parametrize(
    "model",
    [
        SVMClassifier(kernel="linear", C=1.0, probability=True),
        KNNClassifier(n_neighbors=5)
    ]
)
def test_advanced_classification_pipeline(sample_classification_data, model):
    """Test classification pipeline with SVM and KNN classification models."""
    X, y = sample_classification_data
    
    # Split data
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    test_size = int(0.2 * n_samples)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # Train the model
    trainer = Trainer(model, validation_split=0.2)
    trained_model = trainer.train(X_train, y_train)
    
    # Get metrics
    metrics = trainer.get_metrics()
    assert "train_score" in metrics
    assert "val_score" in metrics
    
    # Create predictor
    predictor = Predictor(trained_model)
    result = predictor.predict(X_test, return_probabilities=True)
    assert isinstance(result, dict)
    assert "predictions" in result
    assert "probabilities" in result
    
    y_pred = result["predictions"]
    y_prob = result["probabilities"]
    
    # Evaluate predictions
    evaluator = ClassificationMetrics()
    results = evaluator.evaluate(y_test, y_pred, y_prob)
    
    # Check that all expected metrics are present
    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1" in results
    assert "confusion_matrix" in results
    
    # Check that accuracy is reasonable
    assert results["accuracy"] > 0.7  # Should be decent for linearly separable data
