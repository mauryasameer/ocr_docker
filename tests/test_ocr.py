import os
import pytest
from core.ocr_engine import OCRFactory, PaddleOCREngine
from core.evaluators.evaluator import OCREvaluator

def test_paddle_engine_initialization():
    """Test if PaddleOCREngine can be initialized via factory."""
    engine = OCRFactory.get_engine("paddle")
    assert isinstance(engine, PaddleOCREngine)
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

def test_factory_list():
    """Test if factory lists available engines."""
    engines = OCRFactory.list_available_engines()
    assert "paddle" in engines
    assert "easyocr" in engines
    assert "tesseract" in engines
