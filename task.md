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
- [x] Fixed `PaddleOCREngine`'s constructor for PaddleOCR 3.x's rewritten API (`use_angle_cls`→`use_textline_orientation`, dropped `show_log`, `use_gpu`→`device="cpu"`), pinned `paddlepaddle`/`paddleocr` to known-working versions

## Backlog

- [ ] Add `tests/test_data/` sample images for integration tests
