#!/usr/bin/env python
"""
Script to fix imports in test files.
Replaces 'from ml_library.module' with 'from module'.
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Replace ml_library imports with direct imports in a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace ml_library imports
    modified_content = re.sub(r'from ml_library\.([\w\.]+) import', r'from \1 import', content)
    
    if content != modified_content:
        with open(file_path, 'w') as f:
            f.write(modified_content)
        print(f"Fixed imports in {file_path}")

def process_directory(dir_path):
    """Process all Python files in a directory and its subdirectories."""
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                fix_imports_in_file(file_path)

# Process tests directory
project_root = Path(__file__).parent
tests_path = project_root / 'tests'

if tests_path.exists() and tests_path.is_dir():
    print(f"Processing tests directory...")
    process_directory(tests_path)
else:
    print(f"Tests directory not found.")

print("Test import fixing completed.")
