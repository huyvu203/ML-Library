"""Logistic regression model implementation."""

from typing import Dict, List, Optional, Union

import numpy as np

from ..utils.logger import get_logger
from .base import BaseModel

logger = get_logger()


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation function.
    
    Args:
        z: Input array
        
    Returns:
        Sigmoid of the input
    """
    # Clip to avoid overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


class LogisticRegression(BaseModel):
    """Logistic Regression classifier.
    
    This model optimizes the log-likelihood using gradient descent or can
    use scikit-learn's implementation.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
        penalty: str = "none",
        C: float = 1.0,
        **kwargs
    ) -> None:
        """Initialize the logistic regression model.
        
        Args:
            learning_rate: The learning rate for gradient descent
            n_iterations: Maximum number of iterations for gradient descent
            tol: Tolerance for stopping criterion
            fit_intercept: Whether to calculate the intercept for this model
            penalty: Regularization penalty - 'none', 'l1', or 'l2'
            C: Inverse of regularization strength
            **kwargs: Additional parameters
        """
        super().__init__(
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            tol=tol,
            fit_intercept=fit_intercept,
            penalty=penalty,
            C=C,
            **kwargs
        )
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.C = C
        self.classes_: Optional[np.ndarray] = None
        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        
        if penalty not in ["none", "l1", "l2"]:
            raise ValueError("penalty must be 'none', 'l1', or 'l2'")
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Fit the logistic regression model using gradient descent.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: The fitted model
        """
        X = self._validate_data(X)
        y = np.array(y)
        
        # Get unique classes
        self.classes_ = np.unique(y)
        
        if len(self.classes_) != 2:
            raise ValueError(
                f"This LogisticRegression implementation only supports binary classification. "
                f"Found {len(self.classes_)} classes."
            )
        
        # Map classes to 0 and 1
        y_binary = np.where(y == self.classes_[1], 1, 0)
        # Keep y as 1D array for calculations
        y_binary = y_binary.reshape(-1)
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        if self.fit_intercept:
            # Add a column of ones for the intercept
            X_with_intercept = np.hstack((np.ones((n_samples, 1)), X))
            self.weights_ = np.zeros(n_features + 1)
        else:
            X_with_intercept = X
            self.weights_ = np.zeros(n_features)
        
        # Gradient Descent
        prev_cost = float('inf')
        for i in range(self.n_iterations):
            # Calculate predictions (probability of class 1)
            linear_model = np.dot(X_with_intercept, self.weights_)
            y_pred = sigmoid(linear_model)
            
            # Calculate error
            error = y_pred - y_binary
            
            # Calculate gradients
            gradient = (1/n_samples) * np.dot(X_with_intercept.T, error)
            
            # Add regularization term if specified
            if self.penalty == "l2":
                # L2 regularization (exclude intercept from regularization)
                if self.fit_intercept:
                    reg_term = np.zeros_like(self.weights_)
                    reg_term[1:] = self.weights_[1:] / self.C
                else:
                    reg_term = self.weights_ / self.C
                gradient += reg_term
            elif self.penalty == "l1":
                # L1 regularization (exclude intercept from regularization)
                if self.fit_intercept:
                    reg_term = np.zeros_like(self.weights_)
                    reg_term[1:] = np.sign(self.weights_[1:]) / self.C
                else:
                    reg_term = np.sign(self.weights_) / self.C
                gradient += reg_term
            
            # Update parameters
            self.weights_ -= self.learning_rate * gradient
            
            # Calculate cost (Cross-entropy loss)
            epsilon = 1e-15  # Small value to prevent log(0)
            cost = -(1/n_samples) * np.sum(
                y_binary * np.log(y_pred + epsilon) + 
                (1 - y_binary) * np.log(1 - y_pred + epsilon)
            )
            
            # Add regularization to cost if specified
            if self.penalty == "l2":
                if self.fit_intercept:
                    cost += (0.5 / self.C) * np.sum(self.weights_[1:] ** 2)
                else:
                    cost += (0.5 / self.C) * np.sum(self.weights_ ** 2)
            elif self.penalty == "l1":
                if self.fit_intercept:
                    cost += (1 / self.C) * np.sum(np.abs(self.weights_[1:]))
                else:
                    cost += (1 / self.C) * np.sum(np.abs(self.weights_))
            
            # Check for convergence
            if np.abs(prev_cost - cost) < self.tol:
                logger.debug(f"Converged after {i+1} iterations")
                break
                
            prev_cost = cost
        
        # Extract coefficients and intercept
        if self.fit_intercept:
            self.intercept_ = self.weights_[0]
            self.coefficients_ = self.weights_[1:]
        else:
            self.intercept_ = 0.0
            self.coefficients_ = self.weights_
            
        self._fitted = True
        logger.info(f"LogisticRegression fitted after {i+1} iterations with cost: {cost:.6f}")
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability estimates for both classes.
        
        Args:
            X: Samples of shape (n_samples, n_features)
            
        Returns:
            Probability of classes for each sample of shape (n_samples, 2)
        """
        if not self._fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        if self.coefficients_ is None or self.classes_ is None:
            raise ValueError("Model is not fitted correctly.")
            
        X = self._validate_data(X)
        
        # Calculate probability of class 1
        linear_model = np.dot(X, self.coefficients_) + self.intercept_
        prob_class_1 = sigmoid(linear_model)
        
        # Stack probabilities for class 0 and 1
        return np.column_stack((1 - prob_class_1, prob_class_1))
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.
        
        Args:
            X: Samples of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels of shape (n_samples,)
        """
        if self.classes_ is None:
            raise ValueError("Model is not fitted correctly.")
            
        proba = self.predict_proba(X)
        return np.where(proba[:, 1] >= 0.5, self.classes_[1], self.classes_[0])
        
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the accuracy on the given test data and labels.
        
        Args:
            X: Test samples
            y: True labels
            
        Returns:
            Accuracy score
        """
        if not self._fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
        
    def get_params(self) -> Dict[str, Union[float, int, bool, str]]:
        """Get model parameters.
        
        Returns:
            Dict of model parameters
        """
        return {
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "tol": self.tol,
            "fit_intercept": self.fit_intercept,
            "penalty": self.penalty,
            "C": self.C,
        }
        
    def _validate_data(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Validate the input data.
        
        Args:
            X: Input data
            
        Returns:
            Validated numpy array
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float64)
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return X
