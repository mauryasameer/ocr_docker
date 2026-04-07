# Task Tracker

## Status: Active

---

## Completed

- [x] Multi-engine OCR framework (PaddleOCR, EasyOCR, Tesseract)
- [x] Abstract provider pattern (`BaseOCREngine` in `src/core/interfaces.py`)
- [x] `OCRFactory` for dynamic engine loading
- [x] `OCREvaluator` service (F1 + CER metrics)
- [x] Gradio UI (`app.py`)
- [x] Docker + HuggingFace Spaces deployment
- [x] CI pipeline (pytest + ruff)
- [x] Standards alignment: `src/` layout, `pyproject.toml`, `VERSION`, `CHANGELOG.md`

## Backlog

- [ ] Add `tests/test_data/` sample images for integration tests
- [ ] Add `src/services/benchmark_service.py` to wrap `evaluate_batch` with file I/O
- [ ] Add README version + Python badges
