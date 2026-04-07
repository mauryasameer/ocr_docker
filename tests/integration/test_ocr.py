"""
Integration tests for the full OCR pipeline.
These tests exercise real engine instantiation with mocked heavy deps.
"""
from unittest.mock import patch

import numpy as np
import pytest

from src.providers.paddle_provider import OCRFactory, PaddleOCREngine
from src.services.evaluator import OCREvaluator


def test_paddle_engine_mocked_predict():
    engine = PaddleOCREngine()

    with patch("cv2.imread") as mock_imread, patch("cv2.cvtColor") as mock_cvt:
        mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_cvt.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

        img, text, raw = engine.process_image("fake_path.png")
        assert "Mocked Text" in text
        assert raw[0]["text"] == "Mocked Text"


def test_easyocr_factory_dynamic_loading():
    with patch("src.providers.easyocr_provider.EasyOCREngine") as mock_easy:
        OCRFactory.get_engine("easyocr")
        mock_easy.assert_called_once()


def test_f1_score_calculation():
    evaluator = OCREvaluator()
    assert evaluator.calculate_f1_score("this is a test", "this is a test") == 1.0


def test_cer_calculation():
    evaluator = OCREvaluator()
    assert evaluator.calculate_cer("hello", "hello") == 0.0
