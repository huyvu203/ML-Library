"""Unit tests for SVM models."""

import numpy as np
import pytest

from models import SVMRegressor, SVMClassifier


class TestSVMModels:
    """Test cases for SVM models."""
    
    def test_svm_regressor_init(self):
        """Test that SVMRegressor is initialized correctly."""
        model = SVMRegressor(
            kernel="linear",
            C=0.5,
            epsilon=0.2,
            gamma="auto",
            degree=2,
            random_state=42
        )
        
        assert model.kernel == "linear"
        assert model.C == 0.5
        assert model.epsilon == 0.2
        assert model.gamma == "auto"
        assert model.degree == 2
        assert model.random_state == 42
        assert not model._fitted
    
    def test_svm_classifier_init(self):
        """Test that SVMClassifier is initialized correctly."""
        model = SVMClassifier(
            kernel="linear",
            C=0.5,
            gamma="auto",
            degree=2,
            probability=True,
            random_state=42
        )
        
        assert model.kernel == "linear"
        assert model.C == 0.5
        assert model.gamma == "auto"
        assert model.degree == 2
        assert model.probability is True
        assert model.random_state == 42
        assert not model._fitted


@pytest.mark.parametrize(
    "kernel, C, gamma",
    [
        ("linear", 1.0, "scale"),
        ("rbf", 0.5, "auto"),
        ("poly", 2.0, 0.1)
    ]
)
def test_svm_regressor_get_params(kernel, C, gamma):
    """Test getting parameters from the SVM regressor."""
    model = SVMRegressor(kernel=kernel, C=C, gamma=gamma)
    
    params = model.get_params()
    assert params["kernel"] == kernel
    assert params["C"] == C
    assert params["gamma"] == gamma


@pytest.mark.parametrize(
    "kernel, C, gamma",
    [
        ("linear", 1.0, "scale"),
        ("rbf", 0.5, "auto"),
        ("poly", 2.0, 0.1)
    ]
)
def test_svm_classifier_get_params(kernel, C, gamma):
    """Test getting parameters from the SVM classifier."""
    model = SVMClassifier(kernel=kernel, C=C, gamma=gamma)
    
    params = model.get_params()
    assert params["kernel"] == kernel
    assert params["C"] == C
    assert params["gamma"] == gamma


def test_svm_regressor_fit_predict(sample_regression_data):
    """Test fitting and prediction of SVMRegressor."""
    X, y, _ = sample_regression_data
    
    # Create and fit the model
    model = SVMRegressor(kernel="linear", C=1.0, epsilon=0.1, random_state=42)
    model.fit(X, y)
    
    # Check model is fitted
    assert model._fitted
    assert model.estimator_ is not None
    
    # Test prediction
    y_pred = model.predict(X)
    assert len(y_pred) == len(y)
    
    # Test score method
    score = model.score(X, y)
    assert 0 <= score <= 1


def test_svm_classifier_fit_predict(sample_classification_data):
    """Test fitting and prediction of SVMClassifier."""
    X, y = sample_classification_data
    
    # Create and fit the model
    model = SVMClassifier(kernel="linear", C=1.0, probability=True, random_state=42)
    model.fit(X, y)
    
    # Check model is fitted
    assert model._fitted
    assert model.estimator_ is not None
    assert model.classes_ is not None
    
    # Test prediction
    y_pred = model.predict(X)
    assert len(y_pred) == len(y)
    
    # Test predict_proba
    y_proba = model.predict_proba(X)
    assert y_proba.shape[0] == len(y)
    assert y_proba.shape[1] == len(model.classes_)
    
    # Test score method
    score = model.score(X, y)
    assert 0 <= score <= 1


def test_svm_regressor_not_fitted_error():
    """Test that appropriate error is raised when model is not fitted."""
    model = SVMRegressor()
    X = np.random.rand(10, 5)
    
    with pytest.raises(ValueError, match="Model is not fitted"):
        model.predict(X)


def test_svm_classifier_not_fitted_error():
    """Test that appropriate error is raised when model is not fitted."""
    model = SVMClassifier()
    X = np.random.rand(10, 5)
    
    with pytest.raises(ValueError, match="Model is not fitted"):
        model.predict(X)


def test_svm_classifier_probability_error():
    """Test error when probability is False but predict_proba is called."""
    model = SVMClassifier(probability=False)
    model.estimator_ = "dummy"  # Mock the estimator to pass the _fitted check
    model._fitted = True
    X = np.random.rand(10, 5)
    
    with pytest.raises(ValueError, match="Probability estimates must be enabled"):
        model.predict_proba(X)
