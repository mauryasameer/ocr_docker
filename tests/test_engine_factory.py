import pytest
from core.ocr_engine import OCRFactory, PaddleOCREngine

def test_factory_get_default():
    """Test if the factory returns Paddle by default (if no name provided)."""
    engine = OCRFactory.get_engine()
    assert isinstance(engine, PaddleOCREngine)

def test_factory_invalid_engine():
    """Test if the factory raises ValueError for unknown engines."""
    with pytest.raises(ValueError, match="OCR Engine 'unknown' not found"):
        OCRFactory.get_engine("unknown")

def test_engine_registration():
    """Test if a custom engine can be registered and retrieved."""
    class MockEngine:
        def __init__(self, **kwargs):
            pass
    
    OCRFactory.register_engine("mock", MockEngine)
    assert "mock" in OCRFactory._engines
    
    engine = OCRFactory.get_engine("mock")
    assert isinstance(engine, MockEngine)

def test_available_engines_list():
    """Test if the listing method returns the expected core engines."""
    engines = OCRFactory.list_available_engines()
    assert "paddle" in engines
    assert "easyocr" in engines
    assert "tesseract" in engines
