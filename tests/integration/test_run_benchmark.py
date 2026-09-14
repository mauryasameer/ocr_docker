import json

import pytest

from scripts.run_benchmark import main
from src.core.interfaces import BaseOCREngine
from src.providers.paddle_provider import OCRFactory


class StubOCREngine(BaseOCREngine):
    def __init__(self, **kwargs):
        pass

    def predict(self, image_path):
        return []

    def process_image(self, image_path):
        if "bad" in image_path:
            return None, "Failed to read image.", None
        return None, "stub text", [{"text": "stub text", "confidence": 0.9, "box": []}]


@pytest.fixture
def registered_stub_engine():
    # OCRFactory caches instances by name regardless of kwargs — clear any stale
    # instance from a prior test run before registering fresh (same pattern the
    # repo's own CHANGELOG notes was needed for _instances leakage before).
    OCRFactory._instances.pop("stub", None)
    OCRFactory.register_engine("stub", StubOCREngine)
    yield
    OCRFactory._instances.pop("stub", None)


def test_run_benchmark_writes_report(tmp_path, registered_stub_engine):
    test_cases = [
        {"image_path": "data/gold_standard/whatever1.png", "ground_truth": "stub text"},
        {"image_path": "data/gold_standard/bad_whatever.png", "ground_truth": "stub text"},
    ]
    test_cases_path = tmp_path / "test_cases.json"
    test_cases_path.write_text(json.dumps(test_cases))
    output_path = tmp_path / "report.html"

    exit_code = main(["--engine", "stub", "--test-cases", str(test_cases_path), "--output", str(output_path)])

    assert exit_code == 0
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "Run Info" in content
    assert "stub" in content


def test_run_benchmark_missing_test_cases_file_fails_cleanly(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    exit_code = main(["--test-cases", str(missing_path)])
    assert exit_code == 1


def test_run_benchmark_empty_test_cases_fails_cleanly(tmp_path):
    empty_path = tmp_path / "empty.json"
    empty_path.write_text("[]")
    exit_code = main(["--test-cases", str(empty_path)])
    assert exit_code == 1
