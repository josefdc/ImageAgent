# Contributing to [ProjectName]

Thank you for considering contributing to our AI-powered image editor! This document provides guidelines and instructions for contribution.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Style Guides](#style-guides)
  - [Git Commit Messages](#git-commit-messages)
  - [Python Style Guide](#python-style-guide)
- [Project Structure](#project-structure)

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report. Following these guidelines helps maintainers understand your report and reproduce the issue.

**Before Submitting A Bug Report:**
- Check the [issues](https://github.com/yourusername/image-editor/issues) to see if the problem has already been reported
- Ensure you're using the latest version
- Determine if the problem is actually a bug and not an intended behavior

**How To Submit A Good Bug Report:**
- Use a clear and descriptive title
- Describe the exact steps to reproduce the problem
- Provide specific examples to demonstrate the steps
- Describe the behavior you observed and what you expected to see
- Include screenshots if applicable
- If possible, include system information (OS, Python version, etc.)

### Suggesting Features

This section guides you through submitting a feature suggestion.

**Before Submitting an Enhancement:**
- Check if there is already a similar feature suggestion in [issues](https://github.com/yourusername/image-editor/issues)
- Briefly search to see if the feature has already been discussed

**How To Submit A Good Feature Suggestion:**
- Use a clear and descriptive title
- Provide a detailed description of the suggested enhancement
- Explain why this enhancement would be useful to most users
- List some examples of how this feature would be used
- If possible, include mockups or designs

### Pull Requests

- Fill in the required template
- Do not include issue numbers in the PR title
- Follow the [Python Style Guide](#python-style-guide)
- Include screenshots and animated GIFs in your pull request whenever possible
- Document new code
- End all files with a newline
- Avoid platform-dependent code

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up the required environment variables (e.g., API keys)
5. Run tests to ensure everything is working:
   ```bash
   pytest
   ```

## Style Guides

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests after the first line
- Consider starting the commit message with an applicable emoji:
  - ✨ `:sparkles:` for new features
  - 🐛 `:bug:` for bug fixes
  - 📝 `:memo:` for documentation updates
  - ♻️ `:recycle:` for refactoring code
  - 🧪 `:test_tube:` for adding tests

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use 4 spaces for indentation (not tabs)
- Use docstrings for all public modules, functions, classes, and methods
- Keep line length to a maximum of 88 characters (compatible with Black formatter)
- Run the project's linting tools before submitting PRs:
  ```bash
  # If using flake8
  flake8 .
  
  # If using black
  black .
  ```

## Project Structure

The project is organized as follows:

```text
image-editor/
├── app.py                     # Main Streamlit app entry point
├── pages/                     # Additional app pages
├── agent/                     # LangGraph Agent implementation
├── core/                      # Core image processing & AI service logic
├── state/                     # Streamlit session state management
├── ui/                        # UI components
├── utils/                     # Utility functions and constants
└── tests/                     # Unit/Integration tests
```

When adding new features:
1. Core image processing functions should go in `core/processing.py`
2. New AI service integrations should go in `core/ai_services.py` 
3. Agent tools should be defined in `agent/tools.py`
4. UI components should be placed in the `ui/` directory

---

Thank you for contributing to our project! ❤️