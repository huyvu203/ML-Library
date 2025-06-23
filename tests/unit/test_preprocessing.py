"""Unit tests for preprocessing transformers."""

import numpy as np
import pytest

from preprocessing import StandardScaler, MinMaxScaler


class TestStandardScaler:
    """Test cases for StandardScaler."""
    
    def test_initialization(self):
        """Test that StandardScaler is initialized with correct values."""
        scaler = StandardScaler(with_mean=True, with_std=True)
        
        assert scaler.with_mean is True
        assert scaler.with_std is True
        assert scaler.mean_ is None
        assert scaler.scale_ is None
        assert scaler._fitted is False
        
        scaler = StandardScaler(with_mean=False, with_std=False)
        assert scaler.with_mean is False
        assert scaler.with_std is False
        
    def test_fit(self):
        """Test fitting of StandardScaler."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        fitted_scaler = scaler.fit(X)
        
        assert fitted_scaler is scaler  # Check if it returns self
        assert scaler._fitted is True
        
        # Check computed statistics
        np.testing.assert_almost_equal(scaler.mean_, np.array([3, 4]))
        np.testing.assert_almost_equal(scaler.scale_, np.array([2, 2]))
    
    def test_transform(self):
        """Test transform method of StandardScaler."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        scaler.fit(X)
        
        X_scaled = scaler.transform(X)
        expected_result = np.array([[-1, -1], [0, 0], [1, 1]])
        np.testing.assert_almost_equal(X_scaled, expected_result)
    
    def test_fit_transform(self):
        """Test fit_transform method of StandardScaler."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        
        X_scaled = scaler.fit_transform(X)
        expected_result = np.array([[-1, -1], [0, 0], [1, 1]])
        np.testing.assert_almost_equal(X_scaled, expected_result)
    
    def test_inverse_transform(self):
        """Test inverse_transform method of StandardScaler."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        scaler.fit(X)
        
        X_scaled = scaler.transform(X)
        X_inversed = scaler.inverse_transform(X_scaled)
        np.testing.assert_almost_equal(X_inversed, X)
    
    def test_with_mean_false(self):
        """Test StandardScaler with with_mean=False."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X)
        
        # When with_mean=False, only divide by standard deviation
        expected_result = np.array([[0.5, 1], [1.5, 2], [2.5, 3]])
        np.testing.assert_almost_equal(X_scaled, expected_result)
    
    def test_with_std_false(self):
        """Test StandardScaler with with_std=False."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler(with_std=False)
        X_scaled = scaler.fit_transform(X)
        
        # When with_std=False, only subtract mean
        expected_result = np.array([[-2, -2], [0, 0], [2, 2]])
        np.testing.assert_almost_equal(X_scaled, expected_result)
    
    def test_1d_array(self):
        """Test that 1D arrays are handled correctly."""
        X = np.array([1, 3, 5])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        expected_result = np.array([[-1], [0], [1]])
        np.testing.assert_almost_equal(X_scaled, expected_result)
    
    def test_not_fitted_error(self):
        """Test that appropriate error is raised when not fitted."""
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError, match="not fitted"):
            scaler.transform(X)
            
        with pytest.raises(ValueError, match="not fitted"):
            scaler.inverse_transform(X)


class TestMinMaxScaler:
    """Test cases for MinMaxScaler."""
    
    def test_initialization(self):
        """Test that MinMaxScaler is initialized with correct values."""
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        assert scaler.feature_range == (0, 1)
        assert scaler.min_ is None
        assert scaler.scale_ is None
        assert scaler._fitted is False
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        assert scaler.feature_range == (-1, 1)
    
    def test_fit(self):
        """Test fitting of MinMaxScaler."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = MinMaxScaler()
        fitted_scaler = scaler.fit(X)
        
        assert fitted_scaler is scaler  # Check if it returns self
        assert scaler._fitted is True
        assert scaler._fitted is True
        
        # Check computed statistics
        np.testing.assert_almost_equal(scaler.data_min_, np.array([1, 2]))
        np.testing.assert_almost_equal(scaler.data_max_, np.array([5, 6]))
        np.testing.assert_almost_equal(scaler.data_range_, np.array([4, 4]))
        np.testing.assert_almost_equal(scaler.scale_, np.array([0.25, 0.25]))
        # min_ isn't the minimum value, but a transformation parameter: feature_range[0] - data_min_ * scale_
        np.testing.assert_almost_equal(scaler.min_, np.array([0, 0]))
    
    def test_transform(self):
        """Test transform method of MinMaxScaler."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = MinMaxScaler()
        scaler.fit(X)
        
        X_scaled = scaler.transform(X)
        expected_result = np.array([[0, 0], [0.5, 0.5], [1, 1]])
        np.testing.assert_almost_equal(X_scaled, expected_result)
    
    def test_fit_transform(self):
        """Test fit_transform method of MinMaxScaler."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = MinMaxScaler()
        
        X_scaled = scaler.fit_transform(X)
        expected_result = np.array([[0, 0], [0.5, 0.5], [1, 1]])
        np.testing.assert_almost_equal(X_scaled, expected_result)
    
    def test_inverse_transform(self):
        """Test inverse_transform method of MinMaxScaler."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = MinMaxScaler()
        scaler.fit(X)
        
        X_scaled = scaler.transform(X)
        X_inversed = scaler.inverse_transform(X_scaled)
        np.testing.assert_almost_equal(X_inversed, X)
    
    def test_custom_feature_range(self):
        """Test MinMaxScaler with custom feature_range."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        
        expected_result = np.array([[-1, -1], [0, 0], [1, 1]])
        np.testing.assert_almost_equal(X_scaled, expected_result)
    
    def test_not_fitted_error(self):
        """Test that appropriate error is raised when not fitted."""
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError, match="not fitted"):
            scaler.transform(X)
            
        with pytest.raises(ValueError, match="not fitted"):
            scaler.inverse_transform(X)
