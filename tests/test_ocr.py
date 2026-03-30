import os
import pytest
from core.ocr_engine import OCREngine
from core.evaluators.evaluator import OCREvaluator

def test_ocr_engine_initialization():
    """Test if OCREngine can be initialized."""
    engine = OCREngine(lang='en')
    assert engine is not None
    assert engine.ocr is not None

def test_f1_score_calculation():
    """Test F1 score logic in OCREvaluator."""
    evaluator = OCREvaluator()
    gold = "this is a test"
    pred = "this is a test"
    assert evaluator.calculate_f1_score(gold, pred) == 1.0

    pred_wrong = "this is something else"
    f1 = evaluator.calculate_f1_score(gold, pred_wrong)
    assert 0 < f1 < 1.0

def test_cer_calculation():
    """Test CER logic in OCREvaluator."""
    evaluator = OCREvaluator()
    gold = "hello"
    pred = "hello"
    assert evaluator.calculate_cer(gold, pred) == 0.0

    pred_wrong = "hella"
    cer = evaluator.calculate_cer(gold, pred_wrong)
    assert cer > 0.0
