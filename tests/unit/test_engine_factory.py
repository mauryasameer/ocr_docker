import pytest

from src.providers.paddle_provider import OCRFactory, PaddleOCREngine


def test_factory_get_default():
    engine = OCRFactory.get_engine()
    assert isinstance(engine, PaddleOCREngine)


def test_factory_invalid_engine():
    with pytest.raises(ValueError, match="OCR Engine 'unknown' not found"):
        OCRFactory.get_engine("unknown")


def test_engine_registration():
    class MockEngine:
        def __init__(self, **kwargs):
            pass

    OCRFactory.register_engine("mock", MockEngine)
    assert "mock" in OCRFactory._engines
    assert isinstance(OCRFactory.get_engine("mock"), MockEngine)


def test_available_engines_list():
    engines = OCRFactory.list_available_engines()
    assert "paddle" in engines
    assert "easyocr" in engines
    assert "tesseract" in engines
