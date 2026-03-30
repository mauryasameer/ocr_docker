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
|---|---|---|
| **HuggingFace Space** | Zero-setup browser demo, executive review | [![HF Space](https://img.shields.io/badge/🤗-Open%20Space-blue)](https://huggingface.co/spaces/mauryasameer/OCR) |
| **Local CLI** | Production use, offline, batch processing | See [Getting Started](#-getting-started) below |

---

## 📖 Overview
> "In OCR, 'it reads' is not a valid test result. Accuracy is the only metric."

This framework automates the OCR validation process that professional applications require — turning raw pixel data into structured, audited text evidence. Built to be **modular**, **testable**, and **deployable**.

### **Sample Extraction**
*(The automated PaddleOCR engine rendering bounding boxes for auditing)*

![Sample OCR Result](assets/sample_results.png)

This framework is **production-ready**, optimized for **containarized environments**, ensuring that your data remains secure while providing high-accuracy text extraction.

### **The Problem it Solves**
Many OCR implementations are "black boxes." This framework provides an **audit trail** for every extraction, allowing you to measure **F1 Score** and **CER** (Character Error Rate) against your own gold-standard datasets.

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

### 1. Prerequisites
* **Python 3.9+**
* **Docker (Optional - for containerized run)**
* **Hugging Face Account (For Space deployment)**

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/mauryasameer/ocr_docker.git
cd ocr_docker

# Setup Virtual Environment
python3 -m venv venv
source venv/bin/activate

# Install Core Dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Running a Performance Audit

Use the built-in CLI to evaluate the engine on your gold-standard data:

```bash
# Run the benchmark suite
python3 scripts/run_benchmark.py
```

### 4. Running the Web UI

```bash
python3 app.py
```

---

## 🛠️ The Core Modules

### **1. Accuracy Evaluator (F1 Score)**
Standard text extraction metrics focus on word overlap. This module ensures that every word in the source document is accurately represented, penalizing omissions and false positives.

### **2. Character Error Rate (CER)**
For high-precision documents, we measure the Levenshtein distance at the character level to identify subtle misreadings (e.g., '8' vs 'B').

### **3. Audit Trail Reporting**
Generates a "Committee-Ready" report in `reports/` following every benchmark, providing a timestamped record of model performance — essential for tracking model health under production load.

---

## 🛡️ Compliance & Standards
| Feature | Module | Focus Area |
| --- | --- | --- |
| **Accuracy Audit** | F1 Score | Data Integrity |
| **Fine-Grained Audit** | CER | Character-level precision |
| **Audit Trail** | Reporting | Documentation & Traceability |

---

## ⚖️ License
Distributed under the Apache License 2.0. See `LICENSE` for more information.
Architecture inspired by the `llm_eval` framework for professional ML auditing standards.
