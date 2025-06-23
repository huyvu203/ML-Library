"""Data preprocessing transformers."""

from typing import Optional, Union

import numpy as np

from utils.logger import get_logger

logger = get_logger()


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance.
    
    The standard score of a sample x is calculated as:
        z = (x - u) / s
    where u is the mean of the training samples and s is the standard deviation.
    """
    
    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        """Initialize the scaler.
        
        Args:
            with_mean: If True, center the data before scaling
            with_std: If True, scale the data to unit variance
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self._fitted = False
        
    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Compute the mean and std to be used for later scaling.
        
        Args:
            X: The data used to compute the mean and standard deviation
               used for later scaling along the features axis.
        
        Returns:
            self: The fitted scaler
        """
        X = self._validate_data(X)
        
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1], dtype=X.dtype)
            
        if self.with_std:
            self.scale_ = np.std(X, axis=0, ddof=1)
            # Prevent division by zero
            self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        else:
            self.scale_ = np.ones(X.shape[1], dtype=X.dtype)
            
        self._fitted = True
        logger.debug("StandardScaler fitted")
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features of X according to feature-wise statistics.
        
        Args:
            X: The data to scale
            
        Returns:
            Scaled data
        """
        if not self._fitted:
            raise ValueError("StandardScaler is not fitted yet. Call 'fit' first.")
            
        X = self._validate_data(X)
        
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler is not fitted correctly.")
        
        X_scaled = X.copy()
        if self.with_mean:
            X_scaled -= self.mean_
        if self.with_std:
            X_scaled /= self.scale_
            
        return X_scaled
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it.
        
        Args:
            X: The data to be transformed
            
        Returns:
            Transformed data
        """
        return self.fit(X).transform(X)
        
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Scale back the data to the original representation.
        
        Args:
            X: The data to inverse transform
            
        Returns:
            Inverse transformed data
        """
        if not self._fitted:
            raise ValueError("StandardScaler is not fitted yet. Call 'fit' first.")
            
        X = self._validate_data(X)
        
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler is not fitted correctly.")
        
        X_inversed = X.copy()
        if self.with_std:
            X_inversed *= self.scale_
        if self.with_mean:
            X_inversed += self.mean_
            
        return X_inversed
        
    def _validate_data(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Validate the input data.
        
        Args:
            X: Input data
            
        Returns:
            Validated numpy array
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float64)
        else:
            # Ensure float64 dtype even if it's already an ndarray
            X = X.astype(np.float64)
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return X


class MinMaxScaler:
    """Scale features to a given range.
    
    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between
    zero and one.
    """
    
    def __init__(self, feature_range: tuple = (0, 1)) -> None:
        """Initialize the scaler.
        
        Args:
            feature_range: Desired range of transformed data
        """
        self.feature_range = feature_range
        self.min_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None
        self.data_range_: Optional[np.ndarray] = None
        self._fitted = False
        
    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """Compute the minimum and maximum to be used for later scaling.
        
        Args:
            X: The data used to compute the min and max
               used for later scaling along the features axis.
        
        Returns:
            self: The fitted scaler
        """
        X = self._validate_data(X)
        
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        
        # Handle division by zero
        self.data_range_ = np.where(self.data_range_ == 0, 1.0, self.data_range_)
        
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / self.data_range_
        
        # For our implementation, min_ is a transformation offset parameter
        # It should be 0 when feature_range is (0, 1) and data_min_ is the actual minimum
        self.min_ = np.zeros_like(self.data_min_)
        
        self._fitted = True
        logger.debug("MinMaxScaler fitted")
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features of X according to feature-wise min and max.
        
        Args:
            X: The data to scale
            
        Returns:
            Scaled data
        """
        if not self._fitted:
            raise ValueError("MinMaxScaler is not fitted yet. Call 'fit' first.")
            
        X = self._validate_data(X)
        
        if (self.min_ is None or self.scale_ is None or 
            self.data_min_ is None or self.data_max_ is None):
            raise ValueError("MinMaxScaler is not fitted correctly.")
        
        X_scaled = X.copy()
        # Apply the min-max scaling formula: (X - X_min) * scale = X * scale - X_min * scale
        X_scaled = (X_scaled - self.data_min_) * self.scale_ + self.feature_range[0]
        
        return X_scaled
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it.
        
        Args:
            X: The data to be transformed
            
        Returns:
            Transformed data
        """
        return self.fit(X).transform(X)
        
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Scale back the data to the original representation.
        
        Args:
            X: The data to inverse transform
            
        Returns:
            Inverse transformed data
        """
        if not self._fitted:
            raise ValueError("MinMaxScaler is not fitted yet. Call 'fit' first.")
            
        X = self._validate_data(X)
        
        if (self.min_ is None or self.scale_ is None or 
            self.data_min_ is None or self.data_max_ is None):
            raise ValueError("MinMaxScaler is not fitted correctly.")
        
        X_inversed = X.copy()
        # Invert the scaling: X = (X_scaled - feature_range[0]) / scale + X_min
        X_inversed = (X_inversed - self.feature_range[0]) / self.scale_ + self.data_min_
        
        return X_inversed
        
    def _validate_data(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Validate the input data.
        
        Args:
            X: Input data
            
        Returns:
            Validated numpy array
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float64)
        else:
            # Ensure float64 dtype even if it's already an ndarray
            X = X.astype(np.float64)
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return X
