#!/usr/bin/env python
"""Test script for SVM models."""

from ml_library.models import SVMRegressor, SVMClassifier

def test_init():
    """Test that SVM models are initialized correctly."""
    reg = SVMRegressor(kernel="linear", C=0.5)
    cls = SVMClassifier(kernel="rbf", C=1.0)
    
    print(f"Regressor kernel: {reg.kernel}, C: {reg.C}")
    print(f"Classifier kernel: {cls.kernel}, C: {cls.C}")
    
if __name__ == "__main__":
    print("Testing SVM models...")
    test_init()
    print("Done!")
