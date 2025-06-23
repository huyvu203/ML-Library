"""Unit tests for configuration loading."""

import json
import os

import pytest
import yaml

from config import ConfigLoader


def test_load_yaml(temp_config_path):
    """Test loading configuration from YAML file."""
    config = {
        "model": {
            "type": "linear_regression",
            "params": {
                "learning_rate": 0.01,
                "n_iterations": 1000
            }
        }
    }
    
    config_file = temp_config_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    
    # Test loading
    loaded_config = ConfigLoader.load_yaml(config_file)
    assert loaded_config == config


def test_load_json(temp_config_path):
    """Test loading configuration from JSON file."""
    config = {
        "model": {
            "type": "logistic_regression",
            "params": {
                "learning_rate": 0.05,
                "n_iterations": 500
            }
        }
    }
    
    config_file = temp_config_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f)
    
    # Test loading
    loaded_config = ConfigLoader.load_json(config_file)
    assert loaded_config == config


def test_load_method(temp_config_path):
    """Test the generic load method with both file types."""
    config = {"key": "value"}
    
    yaml_file = temp_config_path / "config.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(config, f)
    
    json_file = temp_config_path / "config.json"
    with open(json_file, "w") as f:
        json.dump(config, f)
    
    # Test loading both file types
    yaml_config = ConfigLoader.load(yaml_file)
    json_config = ConfigLoader.load(json_file)
    
    assert yaml_config == config
    assert json_config == config


def test_load_nonexistent_file():
    """Test that loading a nonexistent file raises an error."""
    with pytest.raises(FileNotFoundError):
        ConfigLoader.load("nonexistent_file.yaml")


def test_load_unsupported_extension(temp_config_path):
    """Test that loading a file with unsupported extension raises an error."""
    txt_file = temp_config_path / "config.txt"
    txt_file.touch()
    
    with pytest.raises(ValueError, match="Unsupported config file extension"):
        ConfigLoader.load(txt_file)


def test_load_invalid_yaml(temp_config_path):
    """Test that loading an invalid YAML file raises an error."""
    invalid_yaml = temp_config_path / "invalid.yaml"
    with open(invalid_yaml, "w") as f:
        f.write("invalid: yaml: content: :")
    
    with pytest.raises(yaml.YAMLError):
        ConfigLoader.load_yaml(invalid_yaml)


def test_load_invalid_json(temp_config_path):
    """Test that loading an invalid JSON file raises an error."""
    invalid_json = temp_config_path / "invalid.json"
    with open(invalid_json, "w") as f:
        f.write("{invalid: json}")
    
    with pytest.raises(json.JSONDecodeError):
        ConfigLoader.load_json(invalid_json)
