import json
import os


def save_json(data: dict, path: str) -> None:
    """Save data to a JSON file, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict | None:
    """Load data from a JSON file, returning None if the file does not exist."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_benchmark_report(metrics: dict) -> str:
    """Format benchmark results as a Markdown string."""
    lines = ["### OCR Benchmark Results\n"]
    for metric, value in metrics.items():
        lines.append(f"- **{metric.capitalize()}**: {value:.4f}")
    return "\n".join(lines)
