import os
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from core.ocr_engine import OCRFactory, PaddleOCREngine
from core.evaluators.evaluator import OCREvaluator

def test_paddle_engine_mocked_predict():
    """Test if PaddleOCREngine correctly handles mocked results."""
    engine = PaddleOCREngine()
    
    with patch("cv2.imread") as mock_imread:
        mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        # Mocking cvtColor since it's used at the end of process_image
        with patch("cv2.cvtColor") as mock_cvt:
            mock_cvt.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
            
            img, text, raw = engine.process_image("fake_path.png")
            assert "Mocked Text" in text
            assert raw[0]['text'] == "Mocked Text"

def test_easyocr_factory_dynamic_loading():
    """Test if EasyOCR engine can be loaded via factory even if not pre-registered."""
    with patch("core.engines.easyocr_engine.EasyOCREngine") as mock_easy:
        # Get engine, this should trigger dynamic import
        engine = OCRFactory.get_engine("easyocr")
        assert engine is not None
        # Verify it was called
        mock_easy.assert_called_once()

def test_f1_score_calculation():
    """Test F1 score logic in OCREvaluator."""
    evaluator = OCREvaluator()
    gold = "this is a test"
    pred = "this is a test"
    assert evaluator.calculate_f1_score(gold, pred) == 1.0

def test_cer_calculation():
    """Test CER logic in OCREvaluator."""
    evaluator = OCREvaluator()
    gold = "hello"
    pred = "hello"
    assert evaluator.calculate_cer(gold, pred) == 0.0
