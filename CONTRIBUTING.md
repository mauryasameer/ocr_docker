# Contributing to OCR-Docker-Framework

## ⚠️ Critical Rules — Read Before Anything Else

> **No code may be merged into `dev` or `main` unless ALL of the following pass:**
> 1. `ruff check .` — zero lint errors
> 2. `pytest tests/unit/ -v` — zero failures
> 3. `pytest tests/integration/ -v` — zero failures
> 4. GitHub Actions CI is green on the PR

Merging code that breaks CI/CD is a blocking violation. Branch protection rules enforce this on `dev` and `main`.

---

## Branching Model

```
feature/* or fix/*
        ↓ PR (CI must pass)
       dev
        ↓ PR (CI must pass)
       main
        ↓ (auto-deploy)
  HuggingFace Space
```

- **Never push directly to `dev` or `main`.**
- All work happens on a `feature/<name>` or `fix/<name>` branch.
- A PR must be opened and all CI checks must pass before merging.
- `dev → main` only happens via PR after CI is green.

---

## Local Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/mauryasameer/ocr_docker.git
cd ocr_docker

# 2. Create a virtual environment (Python 3.11+ recommended for CI parity)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install runtime + dev dependencies
pip install -r requirements.txt
pip install pytest numpy opencv-python-headless ruff
```

---

## Development Workflow

### 1. Create a branch
```bash
git checkout dev
git pull origin dev
git checkout -b fix/your-fix-name      # or feature/your-feature-name
```

### 2. Make your changes
- Core logic lives in `src/`
  - `src/core/` — abstract base interfaces
  - `src/providers/` — OCR engine implementations (paddle, easyocr, tesseract)
  - `src/services/` — evaluator (F1, CER)
  - `src/utils/` — file helpers
- Tests live in `tests/`
  - `tests/unit/` — pure unit tests, no heavy deps, no I/O
  - `tests/integration/` — end-to-end pipeline tests with mocked engines

### 3. Run the full pre-push checklist locally

**You must run all three and see zero failures before pushing:**

```bash
# Lint
ruff check .

# Unit tests
export PYTHONPATH=$(pwd)
pytest tests/unit/ -v --tb=short

# Integration tests
pytest tests/integration/ -v --tb=short
```

If any step fails, fix it before pushing. **Do not push broken code.**

### 4. Push and open a PR to `dev`
```bash
git push origin fix/your-fix-name
gh pr create --base dev --head fix/your-fix-name --title "..." --body "..."
```

### 5. Wait for CI to pass on the PR
- The CI workflow runs lint + tests on Python 3.11 and 3.12.
- **Do not merge until the CI badge on the PR is green.**
- If CI fails, fix the issue on the same branch and push again.

### 6. Merge to `dev`, then promote to `main`
```bash
# Merge feature → dev (after CI green)
gh pr merge <PR_NUMBER> --merge

# Open a second PR: dev → main
gh pr create --base main --head dev --title "chore: promote dev to main"
# Wait for CI green on this PR too, then merge
gh pr merge <PR_NUMBER> --merge
```

---

## Project Structure

```
ocr_docker/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml               # Lint + test on every PR/push
│   │   └── deploy_hf_space.yml  # Deploy to HF (only after CI passes)
├── src/
│   ├── core/interfaces.py       # Abstract BaseOCREngine
│   ├── providers/               # paddle, easyocr, tesseract engines
│   ├── services/evaluator.py    # F1 Score & CER metrics
│   └── utils/file_utils.py      # JSON/file helpers
├── tests/
│   ├── unit/                    # Fast, no-network tests
│   └── integration/             # Mocked end-to-end tests
├── app.py                       # Gradio UI (HuggingFace Spaces)
├── requirements.txt             # Runtime deps
├── pyproject.toml               # Ruff + pytest config
└── CHANGELOG.md                 # Version history
```

---

## Code Style

- Follow **PEP 8**.
- Run `ruff check .` and fix all errors before committing.
- Use meaningful variable names and add docstrings to all public functions and classes.
- Keep test files in `tests/unit/` or `tests/integration/` — never at the repo root.
- Never leave `print()` statements in production code; use `logging`.

---

## Adding a New OCR Engine

1. Create `src/providers/<engine>_provider.py`
2. Implement `BaseOCREngine` (`predict` + `process_image`)
3. Register it in `OCRFactory._engines` inside `paddle_provider.py`
4. Add unit tests in `tests/unit/test_engine_factory.py`
5. Add integration tests in `tests/integration/test_ocr.py`
6. Run the full checklist (lint + unit + integration) before pushing

---

## HuggingFace Deployment

Deployment is **automatic** when a commit reaches `main` — but only **after** the `test` and `lint` jobs in `deploy_hf_space.yml` pass. Deployment will not run if any CI step fails.

Do not manually push to the HuggingFace remote. All syncs happen through the GitHub Actions pipeline.
