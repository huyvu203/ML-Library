#!/bin/bash
cd /home/huy/projects/Production-Ready-ML-Library/ml_library
poetry run python -m pytest --cov=ml_library tests/
