from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from src.providers.paddle_provider import OCRFactory
from src.services.evaluator import OCREvaluator
from src.services.report_service import build_report

DEFAULT_TEST_CASES = "data/gold_standard/accuracy_tests.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_benchmark")
    parser.add_argument("--engine", default="paddle")
    parser.add_argument("--test-cases", default=DEFAULT_TEST_CASES)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    test_cases_path = Path(args.test_cases)
    if not test_cases_path.exists():
        print(f"error: test cases file not found: {test_cases_path}", file=sys.stderr)
        return 1

    test_cases = json.loads(test_cases_path.read_text(encoding="utf-8"))
    if not test_cases:
        print(f"error: no test cases found in {test_cases_path}", file=sys.stderr)
        return 1

    try:
        engine = OCRFactory.get_engine(args.engine)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    evaluator = OCREvaluator()
    results = evaluator.evaluate_batch(engine, test_cases)

    report = build_report(results, args.engine)
    output_path = args.output or f"reports/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report.save(output_path)
    print(f"report written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
