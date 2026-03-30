---
title: OCR
emoji: 📝
colorFrom: pink
colorTo: indigo
sdk: docker
pinned: false
---

# 📝 PaddleOCR Text Extractor

A professional, modular OCR application powered by [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) and [Gradio](https://gradio.app/). This repository features a production-ready architecture inspired by advanced evaluation frameworks, including automated benchmarking and CI/CD integration with Hugging Face Spaces.

## ✨ Key Features

- **Modular Architecture**: Clean separation between the OCR engine (`core/ocr_engine.py`), evaluation logic, and the UI.
- **Automated Evaluation**: Built-in benchmarking suite to measure OCR accuracy (F1 Score and CER) against gold-standard data.
- **Hugging Face Integration**: Automatically syncs to [mauryasameer/OCR](https://huggingface.co/spaces/mauryasameer/OCR) on every push to `main`.
- **Dockerized**: Fully containerized for consistent deployment across environments.
- **Rich Visualization**: Real-time bounding box rendering and confidence scoring.

---

## 🛠️ Project Structure

```text
.
├── core/                   # Core logic
│   ├── ocr_engine.py       # PaddleOCR wrapper
│   ├── evaluators/         # Accuracy and performance metrics
│   └── utils.py            # Helper functions
├── data/
│   └── gold_standard/      # Benchmark images and ground truth
├── scripts/
│   └── run_benchmark.py    # CLI tool for performance auditing
├── tests/                  # Automated unit tests (Pytest)
├── .github/workflows/      # CI/CD (HF Space Sync)
├── app.py                  # Gradio Web Interface
└── Dockerfile              # Container definition
```

---

## 🚀 Quick Start

### Running Locally with Virtual Environment (Recommended)

```bash
# Setup environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Launch the app
python3 app.py
```

### Running with Docker

```bash
docker build -t ocr-app .
docker run -p 7860:7860 ocr-app
```

---

## 📊 Evaluation & Benchmarking

Ensure your OCR performance remains high by running the automated benchmark suite:

```bash
# Run the benchmark audit
python3 scripts/run_benchmark.py
```

This will generate a detailed report in the `reports/` directory, mapping the OCR success rate across your gold-standard test cases.

---

## 📜 CI/CD & Deployment

This repository is configured with a GitHub Action (`.github/workflows/deploy_hf_space.yml`) that automatically deploys the application to Hugging Face Spaces.

**Required Secret**: `HF_TOKEN` must be added to your GitHub repository secrets to enable the automated sync.

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on our development workflow and how to add new test cases.

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
Inspiration drawn from the `llm_eval` framework for professional ML auditing standards.
