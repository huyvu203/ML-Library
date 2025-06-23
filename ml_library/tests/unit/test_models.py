"""Unit tests for the Linear Regression and Logistic Regression models."""

import numpy as np
import pytest

from ml_library.models import LinearRegression, LogisticRegression


def test_linear_regression_initialization():
    """Test that LinearRegression is initialized with correct values."""
    model = LinearRegression(
        learning_rate=0.05,
        n_iterations=2000,
        tol=1e-5,
        fit_intercept=False
    )
    
    assert model.learning_rate == 0.05
    assert model.n_iterations == 2000
    assert model.tol == 1e-5
    assert model.fit_intercept is False
    assert not model._fitted
    
    
def test_linear_regression_fit_predict(sample_regression_data):
    """Test fitting and prediction of LinearRegression."""
    X, y, true_coef = sample_regression_data
    
    # Create and fit the model
    model = LinearRegression(n_iterations=5000, learning_rate=0.01)
    model.fit(X, y)
    
    # Check model is fitted
    assert model._fitted
    assert model.coefficients_ is not None
    assert model.intercept_ is not None
    
    # Check that coefficients are close to the true values
    # This is approximate since we have noise
    tolerance = 0.5  # Allow for some difference due to noise
    np.testing.assert_allclose(model.coefficients_, true_coef, atol=tolerance)
    
    # Test prediction
    y_pred = model.predict(X)
    assert len(y_pred) == len(y)


def test_linear_regression_score(sample_regression_data):
    """Test the score method of LinearRegression."""
    X, y, _ = sample_regression_data
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate score
    score = model.score(X, y)
    
    # R2 should be between 0 and 1 for a reasonable model
    assert 0 <= score <= 1


def test_linear_regression_not_fitted_error():
    """Test that appropriate error is raised when model is not fitted."""
    model = LinearRegression()
    X = np.random.rand(10, 5)
    
    with pytest.raises(ValueError, match="Model is not fitted"):
        model.predict(X)


def test_linear_regression_get_params():
    """Test getting parameters from the model."""
    model = LinearRegression(
        learning_rate=0.1,
        n_iterations=100,
        tol=1e-6,
        fit_intercept=True
    )
    
    params = model.get_params()
    assert params["learning_rate"] == 0.1
    assert params["n_iterations"] == 100
    assert params["tol"] == 1e-6
    assert params["fit_intercept"] is True


def test_logistic_regression_initialization():
    """Test that LogisticRegression is initialized with correct values."""
    model = LogisticRegression(
        learning_rate=0.05,
        n_iterations=2000,
        tol=1e-5,
        fit_intercept=False
    )
    
    assert model.learning_rate == 0.05
    assert model.n_iterations == 2000
    assert model.tol == 1e-5
    assert model.fit_intercept is False
    assert not model._fitted
    
    
def test_logistic_regression_fit_predict(sample_classification_data):
    """Test fitting and prediction of LogisticRegression."""
    X, y = sample_classification_data
    
    # Create and fit the model
    model = LogisticRegression(n_iterations=1000, learning_rate=0.1)
    model.fit(X, y)
    
    # Check model is fitted
    assert model._fitted
    assert model.coefficients_ is not None
    assert model.intercept_ is not None
    
    # Test prediction
    y_pred = model.predict(X)
    assert len(y_pred) == len(y)
    assert set(np.unique(y_pred)).issubset({0, 1})  # Binary classification


def test_logistic_regression_predict_proba(sample_classification_data):
    """Test probability prediction of LogisticRegression."""
    X, y = sample_classification_data
    
    # Create and fit the model
    model = LogisticRegression()
    model.fit(X, y)
    
    # Test probability prediction
    y_proba = model.predict_proba(X)
    assert y_proba.shape == (len(y), 2)  # Binary classification
    assert np.all((0 <= y_proba) & (y_proba <= 1))  # Probabilities are in [0, 1]
    np.testing.assert_allclose(np.sum(y_proba, axis=1), np.ones(len(y)), rtol=1e-10)


def test_logistic_regression_score(sample_classification_data):
    """Test the score method of LogisticRegression."""
    X, y = sample_classification_data
    
    # Create and fit the model
    model = LogisticRegression()
    model.fit(X, y)
    
    # Calculate score
    score = model.score(X, y)
    
    # Accuracy should be between 0 and 1
    assert 0 <= score <= 1


def test_logistic_regression_not_fitted_error():
    """Test that appropriate error is raised when model is not fitted."""
    model = LogisticRegression()
    X = np.random.rand(10, 5)
    
    with pytest.raises(ValueError, match="Model is not fitted"):
        model.predict(X)


def test_logistic_regression_get_params():
    """Test getting parameters from the model."""
    model = LogisticRegression(
        learning_rate=0.1,
        n_iterations=100,
        tol=1e-6,
        fit_intercept=True
    )
    
    params = model.get_params()
    assert params["learning_rate"] == 0.1
    assert params["n_iterations"] == 100
    assert params["tol"] == 1e-6
    assert params["fit_intercept"] is True
