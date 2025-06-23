"""Linear regression model implementation."""

from typing import Dict, Optional, Union

import numpy as np

from utils.logger import get_logger
from .base import BaseModel

logger = get_logger()


class LinearRegression(BaseModel):
    """Linear regression model implementation.
    
    The model fits a linear equation of the form y = X * coefficients + intercept.
    """
    
    def __init__(
        self, 
        fit_intercept: bool = True, 
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tol: float = 1e-4,
        **kwargs
    ) -> None:
        """Initialize the linear regression model.
        
        Args:
            fit_intercept: Whether to calculate the intercept for this model
            learning_rate: The learning rate for gradient descent
            n_iterations: Maximum number of iterations for gradient descent
            tol: Tolerance for stopping criterion
            **kwargs: Additional parameters
        """
        super().__init__(
            fit_intercept=fit_intercept,
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            tol=tol,
            **kwargs
        )
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Fit the linear model using gradient descent.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: The fitted model
        """
        X = self._validate_data(X)
        # Ensure y is one-dimensional for easier calculations
        y = np.array(y).reshape(-1)
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        if self.fit_intercept:
            # Add a column of ones for the intercept
            X_with_intercept = np.hstack((np.ones((n_samples, 1)), X))
            self.weights_ = np.zeros(n_features + 1)
        else:
            X_with_intercept = X
            self.weights_ = np.zeros(n_features)
        
        # For Linear Regression, we can use the analytical solution instead of gradient descent
        # This is more stable and guaranteed to find the global optimum
        iterations_used = 0
        cost = 0.0
        
        try:
            # Try analytical solution using pseudo-inverse (more stable)
            self.weights_ = np.linalg.pinv(X_with_intercept) @ y
            # Calculate final cost for logging
            y_pred = np.dot(X_with_intercept, self.weights_)
            error = y_pred - y
            cost = np.mean(np.square(error))
            logger.debug("Using analytical solution for linear regression")
        except np.linalg.LinAlgError:
            # Fall back to gradient descent if there are numerical issues
            logger.debug("Analytical solution failed, using gradient descent")
            
            # Gradient Descent
            prev_cost = float('inf')
            for iterations_used in range(self.n_iterations):
                # Calculate predictions
                y_pred = np.dot(X_with_intercept, self.weights_)
                
                # Calculate the error
                error = y_pred - y
                
                # Calculate the gradient
                gradient = (1/n_samples) * np.dot(X_with_intercept.T, error)
                
                # Update parameters
                self.weights_ -= self.learning_rate * gradient
                
                # Calculate cost (MSE)
                cost = np.mean(np.square(error))
                
                # Check for convergence
                if np.abs(prev_cost - cost) < self.tol:
                    logger.debug(f"Converged after {iterations_used+1} iterations")
                    break
                    
                prev_cost = cost
            
            # Increment to report the actual number of iterations (0-indexed loop)
            iterations_used += 1
        
        # Extract coefficients and intercept
        if self.fit_intercept:
            self.intercept_ = self.weights_[0]
            self.coefficients_ = self.weights_[1:]
        else:
            self.intercept_ = 0.0
            self.coefficients_ = self.weights_
            
        self._fitted = True
        logger.info(f"LinearRegression fitted with cost: {cost:.6f}")
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear model.
        
        Args:
            X: Samples of shape (n_samples, n_features)
            
        Returns:
            Returns predicted values
        """
        if not self._fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        if self.coefficients_ is None:
            raise ValueError("Model is not fitted correctly.")
            
        X = self._validate_data(X)
        
        return np.dot(X, self.coefficients_) + self.intercept_
        
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction.
        
        Args:
            X: Test samples
            y: True values
            
        Returns:
            R^2 score
        """
        if not self._fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        
        return 1 - u / v
        
    def get_params(self) -> Dict[str, Union[bool, float, int]]:
        """Get model parameters.
        
        Returns:
            Dict of model parameters
        """
        return {
            "fit_intercept": self.fit_intercept,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "tol": self.tol,
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
