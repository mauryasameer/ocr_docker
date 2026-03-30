import sys
from unittest.mock import MagicMock
import numpy as np

# Global mocks for heavyweight OCR libraries to avoid ModuleNotFoundErrors in CI
mock_modules = [
    "paddleocr", 
    "paddleocr.PaddleOCR",
    "paddlex",
    "easyocr",
    "pytesseract"
]

for module in mock_modules:
    sys.modules[module] = MagicMock()

# Specifically handle the OCRResult list-like behavior for PaddleX 3.x
class MockResult(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img = np.zeros((10, 10, 3), dtype=np.uint8)

def mock_predict(*args, **kwargs):
    # PaddleOCREngine expects a list of dictionaries with 'rec_texts'
    res = MockResult({
        'rec_texts': ['Mocked Text'], 
        'rec_scores': [0.99],
        'dt_polys': [[[0, 0], [10, 0]]]
    })
    return [res]

# Inject the mock into the sys.modules
sys.modules["paddleocr"].PaddleOCR.return_value.predict.side_effect = mock_predict
