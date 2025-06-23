"""Complete unit tests for the Predictor class."""

import numpy as np
import os
import pytest
import tempfile
from unittest.mock import Mock, patch

from ml_library.inference.predictor import Predictor
from ml_library.models.base import BaseModel


class TestPredictor:
    """Test cases for the Predictor class."""
    
    def test_initialization(self):
        """Test that Predictor is initialized correctly with a fitted model."""
        # Create a mock model that is fitted
        model = Mock(spec=BaseModel)
        model.check_is_fitted.return_value = None  # Method exists and passes
        model.__class__.__name__ = "MockModel"
        
        predictor = Predictor(model)
        assert predictor.model == model
        
    def test_initialization_with_unfitted_model(self):
        """Test that appropriate error is raised when model is not fitted."""
        model = Mock(spec=BaseModel)
        model.check_is_fitted.side_effect = ValueError("Model is not fitted")
        
        with pytest.raises(ValueError, match="Model must be trained before creating a predictor"):
            Predictor(model)
            
    def test_initialization_with_missing_check_method(self):
        """Test that appropriate error is raised when check_is_fitted is missing."""
        model = Mock(spec=BaseModel)
        model.check_is_fitted.side_effect = AttributeError("No check_is_fitted method")
        
        with pytest.raises(ValueError, match="Model must be trained before creating a predictor"):
            Predictor(model)
    
    def test_predict(self):
        """Test the predict method."""
        # Create a mock model for testing
        model = Mock(spec=BaseModel)
        model.check_is_fitted.return_value = None
        model.predict.return_value = np.array([1, 2, 3])
        
        # Create predictor
        predictor = Predictor(model)
        
        # Test basic prediction
        X = [[1, 2], [3, 4], [5, 6]]
        result = predictor.predict(X)
        
        # Verify model was called with numpy array
        model.predict.assert_called_once()
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))
        
        # Test with numpy array input
        model.predict.reset_mock()
        X_np = np.array([[1, 2], [3, 4], [5, 6]])
        result = predictor.predict(X_np)
        model.predict.assert_called_once()
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))
        
    def test_predict_with_probabilities(self):
        """Test the predict method with probabilities."""
        # Create a mock model for testing
        model = Mock(spec=BaseModel)
        model.check_is_fitted.return_value = None
        model.predict.return_value = np.array([1, 0, 1])
        model.predict_proba.return_value = np.array([
            [0.2, 0.8],
            [0.7, 0.3],
            [0.1, 0.9]
        ])
        
        # Create predictor
        predictor = Predictor(model)
        
        # Test prediction with probabilities
        X = [[1, 2], [3, 4], [5, 6]]
        result = predictor.predict(X, return_probabilities=True)
        
        # Verify model methods were called
        model.predict.assert_called_once()
        model.predict_proba.assert_called_once()
        
        # Check results
        assert isinstance(result, dict)
        assert "predictions" in result
        assert "probabilities" in result
        np.testing.assert_array_equal(result["predictions"], np.array([1, 0, 1]))
        np.testing.assert_array_equal(result["probabilities"], np.array([
            [0.2, 0.8],
            [0.7, 0.3],
            [0.1, 0.9]
        ]))
        
    def test_predict_with_probabilities_handling_attribute_error(self):
        """Test handling of AttributeError when requesting probabilities."""
        # Create a mock model that raises AttributeError
        model = Mock(spec=BaseModel)
        model.check_is_fitted.return_value = None
        model.predict.return_value = np.array([1, 0, 1])
        
        # Make predict_proba raise AttributeError
        model.predict_proba.side_effect = AttributeError("'MockModel' has no attribute 'predict_proba'")
        
        # Create predictor
        predictor = Predictor(model)
        
        # Test prediction with probabilities should still work but return only predictions
        X = [[1, 2], [3, 4], [5, 6]]
        result = predictor.predict(X, return_probabilities=True)
        
        # Should return just predictions, not a dict
        assert not isinstance(result, dict)
        np.testing.assert_array_equal(result, np.array([1, 0, 1]))
        
    def test_predict_with_probabilities_handling_not_implemented_error(self):
        """Test handling of NotImplementedError when requesting probabilities."""
        # Create a mock model that raises NotImplementedError
        model = Mock(spec=BaseModel)
        model.check_is_fitted.return_value = None
        model.predict.return_value = np.array([1, 0, 1])
        
        # Make predict_proba raise NotImplementedError
        model.predict_proba.side_effect = NotImplementedError("Probabilities not implemented")
        
        # Create predictor
        predictor = Predictor(model)
        
        # Test prediction with probabilities should still work but return only predictions
        X = [[1, 2], [3, 4], [5, 6]]
        result = predictor.predict(X, return_probabilities=True)
        
        # Should return just predictions, not a dict
        assert not isinstance(result, dict)
        np.testing.assert_array_equal(result, np.array([1, 0, 1]))
    
    def test_predict_batch(self):
        """Test the predict_batch method."""
        # Create a mock model for testing
        model = Mock(spec=BaseModel)
        model.check_is_fitted.return_value = None
        model.predict.side_effect = [
            np.array([1, 2]),
            np.array([3, 4]),
            np.array([5])
        ]
        
        # Create predictor
        predictor = Predictor(model)
        
        # Test batch prediction
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        result = predictor.predict_batch(X, batch_size=2)
        
        # Check the model was called 3 times (for 3 batches)
        assert model.predict.call_count == 3
        
        # Check model was called with correct batch sizes
        np.testing.assert_array_equal(model.predict.call_args_list[0][0][0], np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(model.predict.call_args_list[1][0][0], np.array([[5, 6], [7, 8]]))
        np.testing.assert_array_equal(model.predict.call_args_list[2][0][0], np.array([[9, 10]]))
        
        # Check combined results
        np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5]))
    
    def test_predict_batch_with_probabilities(self):
        """Test the predict_batch method with probabilities."""
        # Create a mock model for testing
        model = Mock(spec=BaseModel)
        model.check_is_fitted.return_value = None
        model.predict.side_effect = [
            np.array([1, 0]),
            np.array([1, 1])
        ]
        model.predict_proba.side_effect = [
            np.array([[0.2, 0.8], [0.7, 0.3]]),
            np.array([[0.1, 0.9], [0.3, 0.7]])
        ]
        
        # Create predictor
        predictor = Predictor(model)
        
        # Test batch prediction with probabilities
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        result = predictor.predict_batch(X, batch_size=2, return_probabilities=True)
        
        # Check results
        assert isinstance(result, dict)
        assert "predictions" in result
        assert "probabilities" in result
        np.testing.assert_array_equal(result["predictions"], np.array([1, 0, 1, 1]))
        np.testing.assert_array_equal(result["probabilities"], np.array([
            [0.2, 0.8],
            [0.7, 0.3],
            [0.1, 0.9],
            [0.3, 0.7]
        ]))
        
    def test_save_and_load_predictions(self):
        """Test saving and loading predictions."""
        model = Mock(spec=BaseModel)
        model.check_is_fitted.return_value = None
        model.predict.return_value = np.array([1, 2, 3])
        
        predictor = Predictor(model)
        X = [[1, 2], [3, 4], [5, 6]]
        
        # Get predictions
        predictions = predictor.predict(X)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_path = tmp.name
            
        try:
            # Save predictions
            predictor.save_predictions(predictions, temp_path)
            assert os.path.exists(temp_path)
            
            # Load predictions
            loaded_predictions = predictor.load_predictions(temp_path)
            
            # Check loaded predictions
            np.testing.assert_array_equal(predictions, loaded_predictions)
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    def test_save_predictions_with_probabilities(self):
        """Test saving predictions with probabilities."""
        model = Mock(spec=BaseModel)
        model.check_is_fitted.return_value = None
        model.predict.return_value = np.array([1, 0, 1])
        model.predict_proba.return_value = np.array([
            [0.2, 0.8],
            [0.7, 0.3],
            [0.1, 0.9]
        ])
        
        predictor = Predictor(model)
        X = [[1, 2], [3, 4], [5, 6]]
        
        # Get predictions with probabilities
        result = predictor.predict(X, return_probabilities=True)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_path = tmp.name
            
        try:
            # Save predictions
            predictor.save_predictions(result, temp_path)
            assert os.path.exists(temp_path)
            
            # Load predictions
            loaded_result = predictor.load_predictions(temp_path)
            
            # Check loaded predictions
            if isinstance(result, dict) and isinstance(loaded_result, dict):
                np.testing.assert_array_equal(result["predictions"], loaded_result["predictions"])
                np.testing.assert_array_equal(result["probabilities"], loaded_result["probabilities"])
            else:
                np.testing.assert_array_equal(result, loaded_result)
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    def test_load_predictions_nonexistent_file(self):
        """Test that loading from a nonexistent file raises FileNotFoundError."""
        model = Mock(spec=BaseModel)
        model.check_is_fitted.return_value = None
        
        predictor = Predictor(model)
        
        with pytest.raises(FileNotFoundError):
            predictor.load_predictions("nonexistent_file.npy")
