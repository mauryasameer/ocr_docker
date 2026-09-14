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
- [x] `meerax` ecosystem conversion — real benchmark CLI (`scripts/run_benchmark.py`), `meerax`-backed HTML report service, `GOVERNANCE.md`, `docker-compose.yml` for local dev, gold-standard fixtures tracked in git
- [x] README version + Python badges

## Backlog

- [ ] Add `tests/test_data/` sample images for integration tests
- [ ] `PaddleOCREngine`'s constructor passes `show_log`/`use_gpu` kwargs the installed `paddleocr` (3.7.0+) no longer accepts — `requirements.txt` has never pinned a version, so a fresh install can break engine construction; needs pinning to a known-working version or updating the constructor to PaddleOCR 3.x's current API
