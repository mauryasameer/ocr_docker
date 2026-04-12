# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.1.0] - 2026-04-13

### Changed
- Upgraded CI and deploy gate to Python 3.12 (removed 3.11/3.12 matrix)
- Updated ruff `target-version` to `py312` in `pyproject.toml`

### Added
- `code-review-graph==2.2.1` to dev dependencies for local knowledge graph

### Fixed
- Test isolation bug in `test_easyocr_factory_dynamic_loading`: clear `OCRFactory._instances` cache in addition to `_engines` registry

## [1.0.0] - 2026-04-07

### Added
- `src/` layout with `core/interfaces.py`, `providers/`, `services/`, `utils/`
- Abstract `BaseOCREngine` interface in `src/core/interfaces.py`
- Provider implementations moved to `src/providers/`: PaddleOCR, EasyOCR, Tesseract
- `OCREvaluator` service moved to `src/services/evaluator.py`
- `pyproject.toml` with ruff lint config and pytest settings (replaces `pytest.ini`)
- Root-level `conftest.py` for sys.path setup
- `VERSION` and `CHANGELOG.md`
- CI lint job with pinned ruff; test matrix updated to Python 3.11/3.12
- `tests/unit/`, `tests/integration/`, `tests/test_data/` structure

### Changed
- `app.py` imports updated to use `src.providers.paddle_provider`
- `print()` replaced with `logging` throughout library code

[1.0.0]: https://github.com/mauryasameer/ocr_docker/releases/tag/v1.0.0
