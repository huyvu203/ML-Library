"""Unit tests for tree-based models."""

import numpy as np
import pytest

from models import (
    DecisionTreeRegressor,
    DecisionTreeClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
    XGBoostRegressor,
    XGBoostClassifier
)


class TestTreeModels:
    """Test cases for tree-based models."""
    
    def test_decision_tree_regressor_init(self):
        """Test that DecisionTreeRegressor is initialized correctly."""
        model = DecisionTreeRegressor(
            max_depth=5,
            criterion="squared_error",
            random_state=42
        )
        
        assert model.max_depth == 5
        assert model.criterion == "squared_error"
        assert model.random_state == 42
        assert not model._fitted
    
    def test_decision_tree_classifier_init(self):
        """Test that DecisionTreeClassifier is initialized correctly."""
        model = DecisionTreeClassifier(
            max_depth=5,
            criterion="gini",
            random_state=42
        )
        
        assert model.max_depth == 5
        assert model.criterion == "gini"
        assert model.random_state == 42
        assert not model._fitted
    
    def test_random_forest_regressor_init(self):
        """Test that RandomForestRegressor is initialized correctly."""
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        
        assert model.n_estimators == 50
        assert model.max_depth == 5
        assert model.random_state == 42
        assert not model._fitted
    
    def test_random_forest_classifier_init(self):
        """Test that RandomForestClassifier is initialized correctly."""
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        
        assert model.n_estimators == 50
        assert model.max_depth == 5
        assert model.random_state == 42
        assert not model._fitted
    
    def test_xgboost_regressor_init(self):
        """Test that XGBoostRegressor is initialized correctly."""
        model = XGBoostRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        assert model.n_estimators == 50
        assert model.learning_rate == 0.1
        assert model.max_depth == 5
        assert model.random_state == 42
        assert not model._fitted
    
    def test_xgboost_classifier_init(self):
        """Test that XGBoostClassifier is initialized correctly."""
        model = XGBoostClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        assert model.n_estimators == 50
        assert model.learning_rate == 0.1
        assert model.max_depth == 5
        assert model.random_state == 42
        assert not model._fitted


@pytest.mark.parametrize(
    "model_class, model_params",
    [
        (DecisionTreeRegressor, {"max_depth": 5, "random_state": 42}),
        (RandomForestRegressor, {"n_estimators": 10, "max_depth": 5, "random_state": 42}),
        (XGBoostRegressor, {"n_estimators": 10, "max_depth": 3, "random_state": 42})
    ]
)
def test_regressor_fit_predict(sample_regression_data, model_class, model_params):
    """Test that regressor models can fit and predict."""
    X, y, _ = sample_regression_data
    
    # Create and fit the model
    model = model_class(**model_params)
    model.fit(X, y)
    
    # Check model is fitted
    assert model._fitted
    
    # Test prediction shape
    y_pred = model.predict(X)
    assert len(y_pred) == len(y)
    
    # Test score method
    score = model.score(X, y)
    assert 0 <= score <= 1


@pytest.mark.parametrize(
    "model_class, model_params",
    [
        (DecisionTreeClassifier, {"max_depth": 5, "random_state": 42}),
        (RandomForestClassifier, {"n_estimators": 10, "max_depth": 5, "random_state": 42}),
        (XGBoostClassifier, {"n_estimators": 10, "max_depth": 3, "random_state": 42})
    ]
)
def test_classifier_fit_predict(sample_classification_data, model_class, model_params):
    """Test that classifier models can fit and predict."""
    X, y = sample_classification_data
    
    # Create and fit the model
    model = model_class(**model_params)
    model.fit(X, y)
    
    # Check model is fitted
    assert model._fitted
    
    # Test prediction shape
    y_pred = model.predict(X)
    assert len(y_pred) == len(y)
    
    # Test predict_proba
    y_proba = model.predict_proba(X)
    assert y_proba.shape[0] == len(y)
    assert y_proba.shape[1] >= 2  # At least 2 classes
    
    # Test score method
    score = model.score(X, y)
    assert 0 <= score <= 1
