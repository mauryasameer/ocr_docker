---
title: OCR
emoji: 📝
colorFrom: pink
colorTo: indigo
sdk: docker
pinned: false
---

# 🛡️ OCR-Docker-Framework
### **High-Performance Optical Character Recognition & Performance Auditing Suite**
**An opinionated, production-ready OCR engine with automated performance validation.**

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/mauryasameer/OCR)
[![CI](https://github.com/mauryasameer/ocr_docker/actions/workflows/deploy_hf_space.yml/badge.svg)](https://github.com/mauryasameer/ocr_docker/actions)

---

## ⚡ No-Install Quick Start

Don't want to clone the repo? Use one of these:

| Option | Best for | Link |
|:--- |:--- |:--- |
| **HuggingFace Space** | Zero-setup browser demo, executive review | [![HF Space](https://img.shields.io/badge/🤗-Open%20Space-blue)](https://huggingface.co/spaces/mauryasameer/OCR) |
| **Local CLI** | Production use, offline, batch processing | See [Getting Started](#-getting-started) below |

---

## 📖 Overview
> "In OCR, 'it reads' is not a valid test result. Accuracy is the only metric."

This framework automates the OCR validation process that professional applications require — turning raw pixel data into structured, audited text evidence. Built to be **modular**, **testable**, and **deployable**.

### **Sample Report**
*(The automated benchmark report generated after performance auditing)*

```markdown
### OCR Benchmark Results
- **Average F1 Score**: 1.0000
- **Average CER**: 0.0000
```

---

## 🏗️ System Architecture
The framework follows a modular "Engine-Auditor" design:

1.  **OCR Engine Wrapper**: Standardized PaddleOCR interface with robust image handling.
2.  **The Auditor**: A metric-driven suite mapping extractions to ground-truth data.
3.  **Evaluator Modules**:
    *   **F1 Score**: Word-level precision and recall for extraction integrity.
    *   **CER (Character Error Rate)**: Measuring fine-grained recognition success.
4.  **Reporting Engine**: Automated JSON and Markdown generator for audit trails.

---

## 📂 Project Structure
```text
ocr_docker/
├── .github/
│   └── workflows/      # Automated sync to Hugging Face Spaces
├── core/
│   ├── evaluators/     # F1 Score & CER character-level metrics
│   ├── ocr_engine.py   # High-level PaddleOCR orchestrator
│   └── utils.py        # Persistence & reporting utilities
├── data/
│   └── gold_standard/  # Reference images and ground-truth JSON
├── reports/            # Generated audit trails (JSON/Markdown)
├── scripts/
│   └── run_benchmark.py # CLI Entry Point for performance auditing
├── tests/              # Pytest unit testing suite
├── app.py              # Gradio web interface (HuggingFace Spaces)
├── requirements.txt    # Production dependencies
└── requirements-dev.txt # Test-only dependencies
```

---

## 🚀 Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/mauryasameer/ocr_docker.git
cd ocr_docker

# Setup Virtual Environment
python3 -m venv venv
source venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Running a Performance Audit

```bash
# Run the benchmark suite against gold-standard data
python3 scripts/run_benchmark.py
```

### 3. Running Locally (Gradio)

```bash
python3 app.py
```

---

## 🛠️ The Core Modules

### **1. Accuracy Evaluator (F1 Score)**
Standard text extraction metrics focus on word overlap. This module ensures that every word in the source document is accurately represented, penalizing omissions and false positives.

### **2. Character Error Rate (CER)**
For high-precision financial or legal documents, we measure the Levenshtein distance at the character level to identify subtle misreadings (e.g., '8' vs 'B').

### **3. Audit Trail Reporting**
Generates a "Committee-Ready" report in `reports/` following every benchmark, providing a timestamped record of model performance — essential for regulatory compliance.

---

## ⚖️ License
Distributed under the Apache License 2.0. See `LICENSE` for more information.
Architecture inspired by the `llm_eval` framework for professional ML auditing standards.
