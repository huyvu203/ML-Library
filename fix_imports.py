#!/usr/bin/env python
"""
Script to fix relative imports throughout the codebase.
Replaces all instances of 'from ..module' with 'from module'.
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Replace relative imports with absolute imports in a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace relative imports
    modified_content = re.sub(r'from \.\.([\w\.]+) import', r'from \1 import', content)
    
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

# Main modules to process
modules = ['models', 'config', 'inference', 'preprocessing', 'training', 'utils', 'evaluation']
project_root = Path(__file__).parent

for module in modules:
    module_path = project_root / module
    if module_path.exists() and module_path.is_dir():
        print(f"Processing {module}...")
        process_directory(module_path)
    else:
        print(f"Module {module} not found.")

print("Import fixing completed.")
