#!/usr/bin/env python
"""Test script to verify imports are working correctly."""

def test_base_imports():
    """Test importing base models."""
    try:
        from models.base import BaseModel
        print("✅ BaseModel imported successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing BaseModel: {e}")
        return False

def test_linear_models():
    """Test importing linear models."""
    try:
        from models import LinearRegression, LogisticRegression
        print("✅ Linear models imported successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing linear models: {e}")
        return False

def test_tree_models():
    """Test importing tree-based models."""
    try:
        from models import (
            DecisionTreeRegressor, DecisionTreeClassifier,
            RandomForestRegressor, RandomForestClassifier
        )
        print("✅ Tree models imported successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing tree models: {e}")
        return False

def test_boosting_models():
    """Test importing boosting models."""
    try:
        from models import XGBoostRegressor, XGBoostClassifier
        print("✅ XGBoost models imported successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing XGBoost models: {e}")
        return False

def test_svm_models():
    """Test importing SVM models."""
    try:
        from models import SVMRegressor, SVMClassifier
        print("✅ SVM models imported successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing SVM models: {e}")
        return False

def test_knn_models():
    """Test importing KNN models."""
    try:
        from models import KNNRegressor, KNNClassifier
        print("✅ KNN models imported successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing KNN models: {e}")
        return False

def run_all_tests():
    """Run all import tests."""
    print("Running import tests...")
    results = [
        test_base_imports(),
        test_linear_models(),
        test_tree_models(),
        test_boosting_models(),
        test_svm_models(),
        test_knn_models()
    ]
    
    print("\nSummary:")
    if all(results):
        print("✅ All imports successful!")
    else:
        print("❌ Some imports failed. See details above.")
        

if __name__ == "__main__":
    run_all_tests()
