"""Unit tests for KNN models."""

import numpy as np
import pytest

from models import KNNRegressor, KNNClassifier


class TestKNNModels:
    """Test cases for KNN models."""
    
    def test_knn_regressor_init(self):
        """Test that KNNRegressor is initialized correctly."""
        model = KNNRegressor(
            n_neighbors=10,
            weights="distance",
            algorithm="ball_tree",
            leaf_size=40,
            p=1,
            metric="manhattan"
        )
        
        assert model.n_neighbors == 10
        assert model.weights == "distance"
        assert model.algorithm == "ball_tree"
        assert model.leaf_size == 40
        assert model.p == 1
        assert model.metric == "manhattan"
        assert not model._fitted
    
    def test_knn_classifier_init(self):
        """Test that KNNClassifier is initialized correctly."""
        model = KNNClassifier(
            n_neighbors=7,
            weights="distance",
            algorithm="kd_tree",
            leaf_size=20,
            p=1,
            metric="manhattan"
        )
        
        assert model.n_neighbors == 7
        assert model.weights == "distance"
        assert model.algorithm == "kd_tree"
        assert model.leaf_size == 20
        assert model.p == 1
        assert model.metric == "manhattan"
        assert not model._fitted


@pytest.mark.parametrize(
    "n_neighbors, weights, algorithm",
    [
        (3, "uniform", "auto"),
        (5, "distance", "ball_tree"),
        (10, "uniform", "kd_tree")
    ]
)
def test_knn_regressor_get_params(n_neighbors, weights, algorithm):
    """Test getting parameters from the KNN regressor."""
    model = KNNRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
    
    params = model.get_params()
    assert params["n_neighbors"] == n_neighbors
    assert params["weights"] == weights
    assert params["algorithm"] == algorithm


@pytest.mark.parametrize(
    "n_neighbors, weights, algorithm",
    [
        (3, "uniform", "auto"),
        (5, "distance", "ball_tree"),
        (10, "uniform", "kd_tree")
    ]
)
def test_knn_classifier_get_params(n_neighbors, weights, algorithm):
    """Test getting parameters from the KNN classifier."""
    model = KNNClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
    
    params = model.get_params()
    assert params["n_neighbors"] == n_neighbors
    assert params["weights"] == weights
    assert params["algorithm"] == algorithm


def test_knn_regressor_fit_predict(sample_regression_data):
    """Test fitting and prediction of KNNRegressor."""
    X, y, _ = sample_regression_data
    
    # Create and fit the model
    model = KNNRegressor(n_neighbors=3, weights="uniform")
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


def test_knn_classifier_fit_predict(sample_classification_data):
    """Test fitting and prediction of KNNClassifier."""
    X, y = sample_classification_data
    
    # Create and fit the model
    model = KNNClassifier(n_neighbors=3, weights="uniform")
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


def test_knn_regressor_not_fitted_error():
    """Test that appropriate error is raised when model is not fitted."""
    model = KNNRegressor()
    X = np.random.rand(10, 5)
    
    with pytest.raises(ValueError, match="Model is not fitted"):
        model.predict(X)


def test_knn_classifier_not_fitted_error():
    """Test that appropriate error is raised when model is not fitted."""
    model = KNNClassifier()
    X = np.random.rand(10, 5)
    
    with pytest.raises(ValueError, match="Model is not fitted"):
        model.predict(X)


@pytest.mark.parametrize(
    "model_class, kwargs",
    [
        (KNNRegressor, {"n_neighbors": 3}),
        (KNNRegressor, {"n_neighbors": 5, "weights": "distance"}),
        (KNNClassifier, {"n_neighbors": 3}),
        (KNNClassifier, {"n_neighbors": 5, "weights": "distance"})
    ]
)
def test_validate_data(model_class, kwargs):
    """Test data validation in KNN models."""
    model = model_class(**kwargs)
    
    # Test with list
    X_list = [[1, 2, 3], [4, 5, 6]]
    X_array = model._validate_data(X_list)
    
    assert isinstance(X_array, np.ndarray)
    assert X_array.shape == (2, 3)
    
    # Test with numpy array
    X_original = np.array([[7, 8, 9], [10, 11, 12]])
    X_validated = model._validate_data(X_original)
    
    assert isinstance(X_validated, np.ndarray)
    np.testing.assert_array_equal(X_validated, X_original)
