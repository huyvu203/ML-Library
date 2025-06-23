# GitHub Copilot Agent Prompt: Production-Ready ML Library

## Project Goal

Build a **production-ready Python machine learning library** that is clean, modular, well-documented, and fully tested, with a modern CI/CD pipeline.

---

## Environment & Packaging

- Use **Poetry** for dependency and environment management.
- Generate a complete `pyproject.toml` with both runtime and development dependencies.
- Support `poetry install`, `poetry build`, and `poetry publish`.
- 💡 **Important:** All tools, commands, and executions must be done inside the Poetry-managed environment.
  - Use `poetry add` to manage all dependencies (no `pip install`)
  - Use `poetry run` to execute all tools (e.g., `pytest`, `black`, `mypy`, `bandit`, etc.)
  - Assume users will be working inside `poetry shell` or using `poetry run`
  - CI/CD scripts and documentation must follow this workflow consistently

---

## Core Library Features

- Modular pipeline supporting:
  - Data preprocessing
  - Model training (regression and classification)
  - Inference
  - Evaluation
- Configuration management via YAML and JSON.
- Support scikit-learn algorithms like Linear Regression and Logistic Regression.
- Follow the Single Responsibility Principle: each function should do exactly one job.
- Use Python type hints and structured logging with `loguru`.

---

## Testing

- Use `pytest` for unit and integration tests.
- Achieve 90%+ test coverage.
- Use `hypothesis` for property-based testing.
- Include basic performance benchmarks.

---

## Documentation

- Write a detailed `README.md` with setup instructions and usage examples.
- Use PEP257-compliant docstrings and follow PEP8 style.
- Generate documentation with Sphinx.   

---

## Code Quality & Security

- Include the following dev dependencies managed by Poetry:
  - `black` (code formatting)
  - `flake8` (style linting)
  - `mypy` (static type checking)
  - `bandit` (security scanning)
  - `pre-commit` (Git hooks)
- Provide a `.pre-commit-config.yaml` file configured with these tools.
- Install pre-commit hooks in the Poetry environment.

---

## CI/CD Pipeline

- Use GitHub Actions to:
  - Run tests and coverage
  - Run all code quality checks and security scans
  - Generate and publish Sphinx documentation
  - Optionally build and publish the package to PyPI via Poetry

---

## Project Structure

I am not sure what the best folder structure is. Please propose a clean, modular folder structure for this ML library that supports:

- Modular code (preprocessing, training, inference, evaluation)
- Configuration loading
- Tests (unit and integration)
- Documentation with Sphinx
- CI/CD workflows
- Packaging with Poetry

Please generate that folder structure and explain your reasoning.


Please start by generating the folder structure with explanations.
