"""Test SVM model imports and basic initialization."""

from ml_library.models import SVMRegressor, SVMClassifier

def test_svm_imports():
    """Test that SVM models can be imported."""
    assert SVMRegressor is not None
    assert SVMClassifier is not None
    
    # Instantiate models to make sure they work
    reg_model = SVMRegressor(kernel="linear", C=1.0)
    cls_model = SVMClassifier(kernel="linear", C=1.0)
    
    assert reg_model.kernel == "linear"
    assert cls_model.kernel == "linear"
    print("SVM models imported and instantiated successfully!")
