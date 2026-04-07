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
