import pytest

from src.services.evaluator import OCREvaluator


@pytest.fixture
def evaluator():
    return OCREvaluator()


def test_perfect_match(evaluator):
    gold = "The quick brown fox."
    assert evaluator.calculate_f1_score(gold, gold) == 1.0
    assert evaluator.calculate_cer(gold, gold) == 0.0


def test_case_insensitivity(evaluator):
    assert evaluator.calculate_f1_score("Hello World", "hello world") == 1.0
    assert evaluator.calculate_cer("Hello World", "hello world") == 0.0


def test_empty_strings(evaluator):
    assert evaluator.calculate_f1_score("", "") == 1.0
    assert evaluator.calculate_cer("", "") == 0.0


def test_partial_match(evaluator):
    gold = "Welcome to OCR framework"
    pred = "Welcome to framework"
    assert 0 < evaluator.calculate_f1_score(gold, pred) < 1.0
    assert evaluator.calculate_cer(gold, pred) > 0


def test_completely_wrong(evaluator):
    assert evaluator.calculate_f1_score("abc", "xyz") == 0.0
    assert evaluator.calculate_cer("abc", "xyz") == 1.0


def test_evaluate_batch_uses_ground_truth_key(evaluator):
    class GoodEngine:
        def process_image(self, image_path):
            return None, "text", [{"text": "hello world"}]

    test_cases = [{"image_path": "good.png", "ground_truth": "hello world"}]
    result = evaluator.evaluate_batch(GoodEngine(), test_cases)

    assert result["detailed_results"][0]["gold"] == "hello world"
    assert result["detailed_results"][0]["f1"] == 1.0
    assert result["average_f1"] == 1.0


def test_evaluate_batch_isolates_one_bad_entry(evaluator):
    class PartiallyBadEngine:
        def process_image(self, image_path):
            if image_path == "bad.png":
                return None, "Failed to read image.", None
            return None, "text", [{"text": "hello world"}]

    test_cases = [
        {"image_path": "good1.png", "ground_truth": "hello world"},
        {"image_path": "bad.png", "ground_truth": "hello world"},
        {"image_path": "good2.png", "ground_truth": "hello world"},
    ]
    result = evaluator.evaluate_batch(PartiallyBadEngine(), test_cases)

    assert len(result["detailed_results"]) == 3
    good1_entry = next(r for r in result["detailed_results"] if r["image"] == "good1.png")
    bad_entry = next(r for r in result["detailed_results"] if r["image"] == "bad.png")
    good2_entry = next(r for r in result["detailed_results"] if r["image"] == "good2.png")
    assert good1_entry["f1"] == 1.0
    assert bad_entry["pred"] == ""
    assert bad_entry["f1"] == 0.0
    assert good2_entry["f1"] == 1.0
