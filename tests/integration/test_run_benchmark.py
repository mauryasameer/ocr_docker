import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_benchmark import main
from src.core.interfaces import BaseOCREngine
from src.providers.paddle_provider import OCRFactory

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_run_benchmark_malformed_json_fails_cleanly(tmp_path, capsys):
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{not valid json")

    exit_code = main(["--test-cases", str(malformed_path)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "could not parse test cases file" in captured.err
    assert "Traceback" not in captured.err


def test_run_benchmark_unavailable_engine_import_error_fails_cleanly(tmp_path, capsys):
    class BrokenEngine(BaseOCREngine):
        def __init__(self, **kwargs):
            raise ImportError("some-ocr-package not installed. Run: pip install some-ocr-package")

        def predict(self, image_path):
            return []

        def process_image(self, image_path):
            return None, "", []

    OCRFactory._instances.pop("broken", None)
    OCRFactory.register_engine("broken", BrokenEngine)
    try:
        test_cases = [{"image_path": "whatever.png", "ground_truth": "stub text"}]
        test_cases_path = tmp_path / "test_cases.json"
        test_cases_path.write_text(json.dumps(test_cases))

        exit_code = main(["--engine", "broken", "--test-cases", str(test_cases_path)])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "some-ocr-package not installed" in captured.err
        assert "Traceback" not in captured.err
    finally:
        OCRFactory._instances.pop("broken", None)


def test_run_benchmark_all_images_fail_exits_nonzero(tmp_path, registered_stub_engine, capsys):
    test_cases = [
        {"image_path": "data/gold_standard/bad_one.png", "ground_truth": "stub text"},
        {"image_path": "data/gold_standard/bad_two.png", "ground_truth": "stub text"},
    ]
    test_cases_path = tmp_path / "test_cases.json"
    test_cases_path.write_text(json.dumps(test_cases))
    output_path = tmp_path / "report.html"

    exit_code = main(["--engine", "stub", "--test-cases", str(test_cases_path), "--output", str(output_path)])

    assert exit_code == 1
    assert output_path.exists()
    captured = capsys.readouterr()
    assert "all 2 images failed to process" in captured.err


def test_run_benchmark_partial_failure_still_exits_zero(tmp_path, registered_stub_engine):
    test_cases = [
        {"image_path": "data/gold_standard/whatever1.png", "ground_truth": "stub text"},
        {"image_path": "data/gold_standard/bad_whatever.png", "ground_truth": "stub text"},
    ]
    test_cases_path = tmp_path / "test_cases.json"
    test_cases_path.write_text(json.dumps(test_cases))
    output_path = tmp_path / "report.html"

    exit_code = main(["--engine", "stub", "--test-cases", str(test_cases_path), "--output", str(output_path)])

    assert exit_code == 0


def test_run_benchmark_readme_documented_invocation_succeeds(tmp_path):
    """Reproduces the exact command documented in README's "Running a Benchmark"
    section (`python3 -m scripts.run_benchmark ...`) as a real subprocess, the way
    a user copy-pasting the docs would invoke it. This is the seam that let the
    original `ModuleNotFoundError: No module named 'src'` bug through: main()
    called in-process never exercises how `sys.path` is actually populated for a
    real process invocation.

    A stub engine is registered via a `sitecustomize.py` placed on PYTHONPATH so
    this test doesn't depend on a real OCR engine/model being installed — the
    thing under test is whether the `src`/`scripts` import chain resolves for the
    `-m` invocation style, not OCR accuracy.
    """
    sitecustomize_dir = tmp_path / "sitecustomize_dir"
    sitecustomize_dir.mkdir()
    (sitecustomize_dir / "sitecustomize.py").write_text(
        "from src.providers.paddle_provider import OCRFactory\n"
        "from src.core.interfaces import BaseOCREngine\n"
        "\n"
        "\n"
        "class _StubEngine(BaseOCREngine):\n"
        "    def __init__(self, **kwargs):\n"
        "        pass\n"
        "\n"
        "    def predict(self, image_path):\n"
        "        return []\n"
        "\n"
        "    def process_image(self, image_path):\n"
        "        return None, 'stub text', [{'text': 'stub text', 'confidence': 0.9, 'box': []}]\n"
        "\n"
        "\n"
        "OCRFactory.register_engine('stub', _StubEngine)\n"
    )

    test_cases = [{"image_path": "whatever.png", "ground_truth": "stub text"}]
    test_cases_path = tmp_path / "test_cases.json"
    test_cases_path.write_text(json.dumps(test_cases))
    output_path = tmp_path / "report.html"

    env = {**os.environ, "PYTHONPATH": f"{sitecustomize_dir}{os.pathsep}{REPO_ROOT}"}

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_benchmark",
            "--engine",
            "stub",
            "--test-cases",
            str(test_cases_path),
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "ModuleNotFoundError" not in result.stderr
    assert result.returncode == 0
    assert "report written to" in result.stdout
    assert output_path.exists()
