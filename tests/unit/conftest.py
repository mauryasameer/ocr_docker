import sys
from unittest.mock import MagicMock

import numpy as np

# Mock heavyweight OCR libraries so unit tests run without them installed
mock_modules = [
    "paddleocr",
    "paddleocr.PaddleOCR",
    "paddlex",
    "easyocr",
    "pytesseract",
]
for module in mock_modules:
    sys.modules[module] = MagicMock()


class MockResult(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img = np.zeros((10, 10, 3), dtype=np.uint8)


def mock_predict(*args, **kwargs):
    res = MockResult({
        "rec_texts": ["Mocked Text"],
        "rec_scores": [0.99],
        "dt_polys": [[[0, 0], [10, 0]]],
    })
    return [res]


sys.modules["paddleocr"].PaddleOCR.return_value.predict.side_effect = mock_predict
