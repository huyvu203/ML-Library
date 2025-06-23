"""Script to test SVM models."""

from ml_library.models import SVMRegressor, SVMClassifier
import numpy as np
import traceback

def test_svm_regressor():
    """Test SVMRegressor basic functionality."""
    print("\n----- Testing SVMRegressor -----")
    try:
        # Create a simple dataset
        X = np.random.rand(20, 5)
        y = np.random.rand(20)
        
        # Create the model
        print("Creating SVMRegressor...")
        model = SVMRegressor(kernel="linear", C=1.0)
        print(f"Model created: {model}")
        
        # Fit the model
        print("Fitting the model...")
        model.fit(X, y)
        print("Model fitted successfully")
        
        # Predict with the model
        print("Making predictions...")
        y_pred = model.predict(X)
        print(f"Predictions shape: {y_pred.shape}")
        
        print("SVMRegressor test passed!")
        return True
    except Exception as e:
        print(f"SVMRegressor test failed with error: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False

def test_svm_classifier():
    """Test SVMClassifier basic functionality."""
    print("\n----- Testing SVMClassifier -----")
    try:
        # Create a simple dataset
        X = np.random.rand(20, 5)
        y = np.random.randint(0, 2, 20)
        
        # Create the model
        print("Creating SVMClassifier...")
        model = SVMClassifier(kernel="linear", C=1.0, probability=True)
        print(f"Model created: {model}")
        
        # Fit the model
        print("Fitting the model...")
        model.fit(X, y)
        print("Model fitted successfully")
        
        # Predict with the model
        print("Making predictions...")
        y_pred = model.predict(X)
        print(f"Predictions shape: {y_pred.shape}")
        
        # Predict probabilities
        print("Calculating probabilities...")
        y_proba = model.predict_proba(X)
        print(f"Probabilities shape: {y_proba.shape}")
        
        print("SVMClassifier test passed!")
        return True
    except Exception as e:
        print(f"SVMClassifier test failed with error: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing SVM models...")
    test_svm_regressor()
    test_svm_classifier()
