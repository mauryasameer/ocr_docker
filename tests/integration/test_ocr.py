"""
Integration tests for the full OCR pipeline.
These tests exercise real engine instantiation with mocked heavy deps.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.providers.paddle_provider import OCRFactory
from src.services.evaluator import OCREvaluator


def test_paddle_engine_mocked_predict():
    """Test PaddleOCR engine process_image with all heavy deps mocked out."""
    mock_ocr_result = [{
        "rec_texts": ["Mocked Text"],
        "rec_scores": [0.99],
        "dt_polys": [[[0, 0], [10, 0], [10, 10], [0, 10]]],
    }]

    mock_paddle_instance = MagicMock()
    mock_paddle_instance.predict.return_value = mock_ocr_result

    with patch("src.providers.paddle_provider._PADDLE_AVAILABLE", True), \
         patch("src.providers.paddle_provider._PaddleOCR", return_value=mock_paddle_instance), \
         patch("cv2.imread", return_value=np.zeros((10, 10, 3), dtype=np.uint8)), \
         patch("cv2.cvtColor", return_value=np.zeros((10, 10, 3), dtype=np.uint8)), \
         patch("cv2.polylines"):

        from src.providers.paddle_provider import PaddleOCREngine
        engine = PaddleOCREngine.__new__(PaddleOCREngine)
        engine.ocr = mock_paddle_instance

        img, text, raw = engine.process_image("fake_path.png")
        assert "Mocked Text" in text
        assert raw[0]["text"] == "Mocked Text"


def test_easyocr_factory_dynamic_loading():
    with patch("src.providers.easyocr_provider.EasyOCREngine") as mock_easy:
        # Reset registry so easyocr is not cached from a previous test
        OCRFactory._engines.pop("easyocr", None)
        OCRFactory.get_engine("easyocr")
        mock_easy.assert_called_once()


def test_f1_score_calculation():
    evaluator = OCREvaluator()
    assert evaluator.calculate_f1_score("this is a test", "this is a test") == 1.0


def test_cer_calculation():
    evaluator = OCREvaluator()
    assert evaluator.calculate_cer("hello", "hello") == 0.0
