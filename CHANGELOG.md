# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.2.1] - 2026-09-14

### Fixed
- `PaddleOCREngine` (the default and only tested-in-production OCR engine) failed to construct at all on the live HuggingFace Space: `ValueError: Unknown argument: show_log`. PaddleOCR 3.x rewrote its constructor — `show_log`/`use_gpu` were removed entirely, `use_angle_cls` was renamed to `use_textline_orientation` — but `requirements.txt` had never pinned a `paddleocr` version, so a routine environment rebuild silently picked up the breaking 3.x release. Fixed the constructor call and pinned `paddlepaddle==3.3.1`/`paddleocr==3.7.0` to the verified-working versions. Added a real test asserting the exact constructor kwargs, since the existing global `paddleocr` mock in `conftest.py` never validated kwargs and let this regression through undetected.

---

## [1.2.0] - 2026-09-14

### Added
- `scripts/run_benchmark.py` — a real, working CLI for the "Audit Trail Reporting" feature the README has described since v1.0.0 but that was never wired up (`format_benchmark_report()`/`save_json()`/`load_json()` were dead code with zero call sites). Runs the existing `OCREvaluator` against `data/gold_standard/accuracy_tests.json`, rendering results via a new `src/services/report_service.py` using `meerax.report.ReportBuilder`.
- `GOVERNANCE.md` — intended use, explainability boundary, fairness scope (not applicable — no demographic data), audit trail, regulatory framing.
- `docker-compose.yml` for local one-click dev use (`docker compose up --build`), alongside the existing HuggingFace-Spaces-specific `Dockerfile`, which is unchanged.
- `meerax==1.10.2` added to `requirements-dev.txt` (not `requirements.txt` — it's a benchmark/test-only dependency, never deployed to the live HF Space).

### Fixed
- `OCREvaluator.evaluate_batch()` read `case["text"]` for gold-standard text, but the real fixture format (`data/gold_standard/accuracy_tests.json`) uses `"ground_truth"` — a latent key mismatch, never caught because nothing called this method against real data before this release. Also added per-entry failure isolation so one unreadable image no longer crashes the whole batch.
- Stale `pyproject.toml` ruff `exclude` list referencing a `core/` directory removed in v1.0.0.
- `.DS_Store` was untracked but not gitignored.
- README's project-structure tree and version badge had drifted from the actual repo state.

---

## [1.1.0] - 2026-04-13

### Changed
- Upgraded CI and deploy gate to Python 3.12 (removed 3.11/3.12 matrix)
- Updated ruff `target-version` to `py312` in `pyproject.toml`

### Added
- `code-review-graph==2.2.1` to dev dependencies for local knowledge graph

### Fixed
- Test isolation bug in `test_easyocr_factory_dynamic_loading`: clear `OCRFactory._instances` cache in addition to `_engines` registry

---

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
