# Contributing to OCR Docker

We welcome contributions! Please follow these steps to contribute:

## Local Setup

1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

## Development Workflow

1.  **Restructure**: Core logic resides in the `core/` directory.
2.  **Modularize**: Keep processing logic in `core/ocr_engine.py` and evaluation logic in `core/evaluators/`.
3.  **Test**: Add tests in the `tests/` directory and run them using `pytest`.
    ```bash
    pytest
    ```
4.  **Evaluate**: Before submitting a PR, run the benchmark to ensure no regressions in OCR accuracy.
    ```bash
    python3 scripts/run_benchmark.py
    ```

## Hugging Face Deployment

The project is automatically synced to Hugging Face Spaces on every push to the `main` branch. Ensure your changes are compatible with the Gradio environment.

## Code Style

- Follow PEP 8 guidelines.
- Use meaningful variable names and add docstrings to functions and classes.
