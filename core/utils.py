# core/utils.py

import json
import os

def save_json(data, path):
    """Saves data to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_json(path):
    """Loads data from a JSON file."""
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_benchmark_report(metrics):
    """Formats benchmark results for display."""
    report = "### OCR Benchmark Results\n\n"
    for metric, value in metrics.items():
        report += f"- **{metric.capitalize()}**: {value:.4f}\n"
    return report
