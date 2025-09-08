# Contributing to FXorcist

## Welcome! 🚀

We're thrilled that you're interested in contributing to FXorcist, an event-driven Forex trading research platform. This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can You Contribute?

### 🐛 Reporting Bugs
- Use the GitHub Issues section
- Provide a clear and descriptive title
- Describe the exact steps to reproduce the problem
- Include system details and Python version

### 🌟 Feature Requests
- Open a GitHub Issue
- Describe the feature and its potential benefits
- Provide context on how it fits the project's goals

## Development Process

### 🛠 Setup
1. Fork the repository
2. Clone your fork
3. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
4. Install development dependencies
   ```bash
   pip install -e .[dev]
   ```

### 🧪 Running Tests
```bash
pytest
ruff check .
black --check .
mypy fxorcist
```

### 📝 Commit Guidelines
- Use conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, etc.
- Keep commits focused and atomic
- Write clear, descriptive commit messages

### 🔀 Pull Request Process
1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Write or update tests
4. Ensure all tests pass
5. Submit a Pull Request with:
   - Clear title and description
   - Reference any related issues
   - Describe the changes and rationale

### 🤝 Code Review Process
- All submissions require review
- We use GitHub's review system
- Expect feedback and potential requested changes

## Development Workflow

### Branching Strategy
- `main`: Stable release branch
- `develop`: Integration branch for features
- Feature branches: `feature/` prefix
- Bugfix branches: `bugfix/` prefix

### Tools and Standards
- Python 3.10+
- Type hints required
- Black for formatting
- Ruff for linting
- Pytest for testing
- Mypy for type checking

## Performance and Optimization

### 📊 Benchmarking
- Include performance benchmarks for significant changes
- Use `pytest-benchmark` for measuring performance

### 🚀 Optimization Guidelines
- Prefer vectorized operations
- Use efficient data structures
- Minimize memory allocations
- Consider computational complexity

## Machine Learning Contributions

### 🤖 Strategy Development
- Implement strategies as subclasses of `BaseStrategy`
- Include docstrings explaining strategy logic
- Provide example configurations
- Write unit tests covering strategy behavior

## Documentation

- Update docstrings for new features
- Keep README and other docs in sync with code changes
- Include type hints and clear function descriptions

## Financial Disclaimer

Contributions related to trading strategies must include:
- Backtesting results
- Risk assessment
- Performance metrics
- Limitations and potential biases

## Questions?

If you have questions, please:
- Check existing documentation
- Open a GitHub Issue
- Join our community discussions

Thank you for contributing to FXorcist! 🌍💹