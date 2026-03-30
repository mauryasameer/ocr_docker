import pytest
from core.evaluators.evaluator import OCREvaluator

@pytest.fixture
def evaluator():
    return OCREvaluator()

def test_perfect_match(evaluator):
    gold = "The quick brown fox."
    pred = "The quick brown fox."
    assert evaluator.calculate_f1_score(gold, pred) == 1.0
    assert evaluator.calculate_cer(gold, pred) == 0.0

def test_case_insensitivity(evaluator):
    gold = "Hello World"
    pred = "hello world"
    # F1 score calculation in current evaluator might be case-sensitive or not
    # but CER usually is.
    assert evaluator.calculate_f1_score(gold, pred) == 1.0
    assert evaluator.calculate_cer(gold, pred) == 0.0

def test_empty_strings(evaluator):
    assert evaluator.calculate_f1_score("", "") == 1.0
    assert evaluator.calculate_cer("", "") == 0.0

def test_partial_match(evaluator):
    gold = "Welcome to OCR framework"
    pred = "Welcome to framework"
    f1 = evaluator.calculate_f1_score(gold, pred)
    cer = evaluator.calculate_cer(gold, pred)
    assert 0 < f1 < 1.0
    assert cer > 0

def test_completely_wrong(evaluator):
    gold = "abc"
    pred = "xyz"
    # Over 1.0 because every character is wrong
    assert evaluator.calculate_f1_score(gold, pred) == 0.0
    assert evaluator.calculate_cer(gold, pred) == 1.0
