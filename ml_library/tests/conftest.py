"""Test configuration for pytest."""

import os
import sys
from pathlib import Path

import pytest

# Add the parent directory to sys.path to make imports work
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import your library here for fixtures
from ml_library.models import LinearRegression, LogisticRegression
from ml_library.utils.logger import setup_logger

# Configure logging for tests
setup_logger(level="DEBUG")


@pytest.fixture
def sample_regression_data():
    """Generate sample regression data for testing."""
    import numpy as np
    np.random.seed(42)
    
    X = np.random.rand(100, 5)
    true_coef = np.array([3.5, 1.7, -4.2, 2.1, -1.3])
    y = X @ true_coef + np.random.normal(0, 0.5, 100)
    
    return X, y, true_coef


@pytest.fixture
def sample_classification_data():
    """Generate sample classification data for testing."""
    import numpy as np
    np.random.seed(42)
    
    # Generate a simple linearly separable dataset
    X = np.random.rand(100, 2) * 10 - 5  # Values between -5 and 5
    # Class determined by x + y > 0
    y = np.where(X.sum(axis=1) > 0, 1, 0)
    
    return X, y


@pytest.fixture
def linear_regression_model():
    """Create a basic linear regression model for testing."""
    return LinearRegression(
        learning_rate=0.01, 
        n_iterations=1000, 
        tol=1e-4,
    )


@pytest.fixture
def logistic_regression_model():
    """Create a basic logistic regression model for testing."""
    return LogisticRegression(
        learning_rate=0.01,
        n_iterations=1000,
        tol=1e-4,
        fit_intercept=True,
    )


@pytest.fixture
def temp_config_path(tmp_path):
    """Create a temporary directory for config files."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    return config_dir
