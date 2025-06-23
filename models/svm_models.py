"""Support Vector Machine model implementations."""

from typing import Dict, Optional, Union, Any

import numpy as np

from utils.logger import get_logger
from .base import BaseModel

logger = get_logger()


class SVMRegressor(BaseModel):
    """Support Vector Machine Regressor.
    
    This model implements Support Vector Regression (SVR) for regression tasks.
    """
    
    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1,
        gamma: Union[str, float] = "scale",
        degree: int = 3,
        shrinking: bool = True,
        tol: float = 1e-3,
        random_state: Optional[int] = None,
        **kwargs
    ) -> None:
        """Initialize the SVM regressor.
        
        Args:
            kernel: Specifies the kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parameter
            epsilon: Epsilon in the epsilon-SVR model
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            degree: Degree of the polynomial kernel
            shrinking: Whether to use the shrinking heuristic
            tol: Tolerance for stopping criterion
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters
        """
        super().__init__(
            kernel=kernel,
            C=C,
            epsilon=epsilon,
            gamma=gamma,
            degree=degree,
            shrinking=shrinking,
            tol=tol,
            random_state=random_state,
            **kwargs
        )
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.degree = degree
        self.shrinking = shrinking
        self.tol = tol
        self.random_state = random_state
        self.estimator_ = None
        
        # Import scikit-learn implementation
        try:
            from sklearn.svm import SVR
            self.SVR = SVR
        except ImportError:
            logger.error("scikit-learn is required for SVMRegressor")
            raise ImportError(
                "scikit-learn is not installed. "
                "Install it with 'poetry add scikit-learn' or 'pip install scikit-learn'"
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMRegressor":
        """Fit the SVM regression model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: The fitted model
        """
        X = self._validate_data(X)
        y = np.array(y)
        
        logger.debug(
            f"Fitting SVMRegressor with kernel={self.kernel}, C={self.C}, "
            f"epsilon={self.epsilon}, gamma={self.gamma}"
        )
        
        # Create and fit the scikit-learn model
        # SVR in scikit-learn doesn't support random_state parameter
        self.estimator_ = self.SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma,
            degree=self.degree,
            shrinking=self.shrinking,
            tol=self.tol
        )
        
        self.estimator_.fit(X, y)
        self._fitted = True
        
        n_support_vectors = self.estimator_.support_vectors_.shape[0]
        logger.info(
            f"SVMRegressor fitted successfully with {n_support_vectors} support vectors"
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the SVM regression model.
        
        Args:
            X: Samples of shape (n_samples, n_features)
            
        Returns:
            Predicted values
        """
        if not self._fitted or self.estimator_ is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        X = self._validate_data(X)
        return self.estimator_.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction.
        
        Args:
            X: Test samples
            y: True values
            
        Returns:
            R^2 score
        """
        if not self._fitted or self.estimator_ is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        X = self._validate_data(X)
        y = np.array(y)
        
        return self.estimator_.score(X, y)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dict of model parameters
        """
        return {
            "kernel": self.kernel,
            "C": self.C,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "degree": self.degree,
            "shrinking": self.shrinking,
            "tol": self.tol,
            "random_state": self.random_state,
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
            
        return X


class SVMClassifier(BaseModel):
    """Support Vector Machine Classifier.
    
    This model implements Support Vector Classification (SVC) for classification tasks.
    """
    
    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: Union[str, float] = "scale",
        degree: int = 3,
        probability: bool = True,
        shrinking: bool = True,
        tol: float = 1e-3,
        class_weight: Optional[Union[dict, str]] = None,
        random_state: Optional[int] = None,
        **kwargs
    ) -> None:
        """Initialize the SVM classifier.
        
        Args:
            kernel: Specifies the kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            degree: Degree of the polynomial kernel
            probability: Whether to enable probability estimates
            shrinking: Whether to use the shrinking heuristic
            tol: Tolerance for stopping criterion
            class_weight: Weights associated with classes
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters
        """
        super().__init__(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            probability=probability,
            shrinking=shrinking,
            tol=tol,
            class_weight=class_weight,
            random_state=random_state,
            **kwargs
        )
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.probability = probability
        self.shrinking = shrinking
        self.tol = tol
        self.class_weight = class_weight
        self.random_state = random_state
        self.estimator_ = None
        self.classes_ = None
        
        # Import scikit-learn implementation
        try:
            from sklearn.svm import SVC
            self.SVC = SVC
        except ImportError:
            logger.error("scikit-learn is required for SVMClassifier")
            raise ImportError(
                "scikit-learn is not installed. "
                "Install it with 'poetry add scikit-learn' or 'pip install scikit-learn'"
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        """Fit the SVM classification model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: The fitted model
        """
        X = self._validate_data(X)
        y = np.array(y)
        
        logger.debug(
            f"Fitting SVMClassifier with kernel={self.kernel}, C={self.C}, "
            f"gamma={self.gamma}"
        )
        
        # Create and fit the scikit-learn model
        self.estimator_ = self.SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            probability=self.probability,
            shrinking=self.shrinking,
            tol=self.tol,
            class_weight=self.class_weight,
            random_state=self.random_state
        )
        
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        self._fitted = True
        
        n_support_vectors = self.estimator_.support_vectors_.shape[0]
        n_classes = len(self.classes_)
        logger.info(
            f"SVMClassifier fitted successfully with {n_support_vectors} support vectors "
            f"for {n_classes} classes"
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.
        
        Args:
            X: Samples of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels
        """
        if not self._fitted or self.estimator_ is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        X = self._validate_data(X)
        return self.estimator_.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X.
        
        Args:
            X: Samples of shape (n_samples, n_features)
            
        Returns:
            Probability of each class for each sample
        """
        if not self._fitted or self.estimator_ is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        if not self.probability:
            raise ValueError(
                "Probability estimates must be enabled to use predict_proba. "
                "Set probability=True when initializing the model."
            )
            
        X = self._validate_data(X)
        return self.estimator_.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the accuracy on the given test data and labels.
        
        Args:
            X: Test samples
            y: True labels
            
        Returns:
            Accuracy score
        """
        if not self._fitted or self.estimator_ is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        X = self._validate_data(X)
        y = np.array(y)
        
        return self.estimator_.score(X, y)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dict of model parameters
        """
        return {
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "degree": self.degree,
            "probability": self.probability,
            "shrinking": self.shrinking,
            "tol": self.tol,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
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
            
        return X
