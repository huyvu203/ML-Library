#!/usr/bin/env python
"""Test script for SVM models."""

import numpy as np
from ml_library.models import SVMRegressor, SVMClassifier

def test_svm_regressor_init():
    """Test that SVMRegressor is initialized correctly."""
    try:
        model = SVMRegressor(kernel="linear", C=0.5, epsilon=0.2, gamma="auto", degree=2)
        print(f"✅ SVMRegressor initialized: kernel={model.kernel}, C={model.C}")
        return True
    except Exception as e:
        print(f"❌ SVMRegressor initialization failed: {e}")
        return False

def test_svm_classifier_init():
    """Test that SVMClassifier is initialized correctly."""
    try:
        model = SVMClassifier(kernel="linear", C=0.5, gamma="auto", degree=2, probability=True)
        print(f"✅ SVMClassifier initialized: kernel={model.kernel}, C={model.C}")
        return True
    except Exception as e:
        print(f"❌ SVMClassifier initialization failed: {e}")
        return False

def test_svm_regressor_fit_predict():
    """Test SVMRegressor fit and predict with simple data."""
    try:
        # Generate simple data
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([2, 4, 6, 8])  # Linear relationship y = x₁
        
        # Create and fit model
        model = SVMRegressor(kernel="linear")
        model.fit(X, y)
        print("✅ SVMRegressor fit successful")
        
        # Test prediction
        X_test = np.array([[9, 10]])
        y_pred = model.predict(X_test)
        print(f"✅ SVMRegressor predict successful: X_test={X_test}, y_pred={y_pred}")
        return True
    except Exception as e:
        print(f"❌ SVMRegressor fit/predict failed: {e}")
        return False

def test_svm_classifier_fit_predict():
    """Test SVMClassifier fit and predict with simple data."""
    try:
        # Generate simple data
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])  # Binary classification
        
        # Create and fit model
        model = SVMClassifier(kernel="linear", probability=True)
        model.fit(X, y)
        print("✅ SVMClassifier fit successful")
        
        # Test prediction
        X_test = np.array([[5, 6]])
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        print(f"✅ SVMClassifier predict successful: X_test={X_test}, y_pred={y_pred}")
        print(f"✅ SVMClassifier predict_proba successful: probabilities={y_proba}")
        return True
    except Exception as e:
        print(f"❌ SVMClassifier fit/predict failed: {e}")
        return False

def run_tests():
    """Run all SVM tests."""
    print("\nRunning SVM model tests...\n")
    results = [
        test_svm_regressor_init(),
        test_svm_classifier_init(),
        test_svm_regressor_fit_predict(),
        test_svm_classifier_fit_predict()
    ]
    
    print("\nSummary:")
    if all(results):
        print("✅ All SVM tests passed!")
    else:
        print("❌ Some SVM tests failed. See details above.")
    

if __name__ == "__main__":
    run_tests()
