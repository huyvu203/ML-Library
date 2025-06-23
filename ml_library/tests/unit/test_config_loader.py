"""Unit tests for configuration loading."""

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from ml_library.config.loader import ConfigLoader


class TestConfigLoader:
    """Test cases for the ConfigLoader class."""
    
    def test_load_yaml(self):
        """Test loading a YAML configuration file."""
        # Create a temporary YAML config file
        config = {"model": {"name": "test", "params": {"alpha": 0.1}}}
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            yaml_path = tmp.name
        
        try:
            # Load the config
            loaded_config = ConfigLoader.load_yaml(yaml_path)
            
            # Verify the loaded config
            assert loaded_config == config
            assert loaded_config["model"]["name"] == "test"
            assert loaded_config["model"]["params"]["alpha"] == 0.1
        finally:
            # Clean up
            if os.path.exists(yaml_path):
                os.remove(yaml_path)
    
    def test_load_json(self):
        """Test loading a JSON configuration file."""
        # Create a temporary JSON config file
        config = {"model": {"name": "test", "params": {"alpha": 0.1}}}
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            json.dump(config, tmp)
            json_path = tmp.name
        
        try:
            # Load the config
            loaded_config = ConfigLoader.load_json(json_path)
            
            # Verify the loaded config
            assert loaded_config == config
            assert loaded_config["model"]["name"] == "test"
            assert loaded_config["model"]["params"]["alpha"] == 0.1
        finally:
            # Clean up
            if os.path.exists(json_path):
                os.remove(json_path)
    
    def test_load_yaml_with_path_object(self):
        """Test loading a YAML configuration file using a Path object."""
        # Create a temporary YAML config file
        config = {"model": {"name": "test", "params": {"alpha": 0.1}}}
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            yaml_path = Path(tmp.name)
        
        try:
            # Load the config
            loaded_config = ConfigLoader.load_yaml(yaml_path)
            
            # Verify the loaded config
            assert loaded_config == config
        finally:
            # Clean up
            if os.path.exists(yaml_path):
                os.remove(yaml_path)
    
    def test_load_nonexistent_yaml_file(self):
        """Test that loading a nonexistent YAML file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            ConfigLoader.load_yaml("nonexistent_file.yaml")
    
    def test_load_invalid_yaml_file(self):
        """Test that loading an invalid YAML file raises YAMLError."""
        # Create a temporary file with invalid YAML content
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp.write(b"invalid: yaml: content: - [")
            yaml_path = tmp.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                ConfigLoader.load_yaml(yaml_path)
        finally:
            # Clean up
            if os.path.exists(yaml_path):
                os.remove(yaml_path)
    
    def test_load_nonexistent_json_file(self):
        """Test that loading a nonexistent JSON file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            ConfigLoader.load_json("nonexistent_file.json")
    
    def test_load_invalid_json_file(self):
        """Test that loading an invalid JSON file raises JSONDecodeError."""
        # Create a temporary file with invalid JSON content
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(b"{invalid json content")
            json_path = tmp.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                ConfigLoader.load_json(json_path)
        finally:
            # Clean up
            if os.path.exists(json_path):
                os.remove(json_path)
    
    def test_load_auto_yaml(self):
        """Test auto-detection of YAML files."""
        config = {"model": {"name": "test"}}
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            yaml_path = tmp.name
        
        try:
            loaded_config = ConfigLoader.load(yaml_path)
            assert loaded_config == config
        finally:
            if os.path.exists(yaml_path):
                os.remove(yaml_path)
    
    def test_load_auto_yml(self):
        """Test auto-detection of YML files."""
        config = {"model": {"name": "test"}}
        
        with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as tmp:
            yaml.dump(config, tmp)
            yml_path = tmp.name
        
        try:
            loaded_config = ConfigLoader.load(yml_path)
            assert loaded_config == config
        finally:
            if os.path.exists(yml_path):
                os.remove(yml_path)
    
    def test_load_auto_json(self):
        """Test auto-detection of JSON files."""
        config = {"model": {"name": "test"}}
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            json.dump(config, tmp)
            json_path = tmp.name
        
        try:
            loaded_config = ConfigLoader.load(json_path)
            assert loaded_config == config
        finally:
            if os.path.exists(json_path):
                os.remove(json_path)
    
    def test_load_unsupported_extension(self):
        """Test that loading a file with an unsupported extension raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported config file extension"):
            ConfigLoader.load("config.txt")
