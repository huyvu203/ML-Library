## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

- **CI Workflow**: Runs tests, code quality checks, and security scans on every push and pull request
- **Documentation Workflow**: Builds and publishes documentation to GitHub Pages
- **Publish Workflow**: Builds and publishes the package to PyPI when a new version is tagged

### GitHub Actions Workflows

- `ci.yml`: Runs tests, coverage, and code quality checks
- `docs.yml`: Builds and deploys documentation
- `publish.yml`: Publishes the package to PyPI

### Setting up GitHub Secrets

To enable all CI/CD features, set up the following GitHub secrets:

- `PYPI_API_TOKEN`: Your PyPI API token for publishing packages
- `SONAR_TOKEN`: SonarCloud API token for code quality analysis
