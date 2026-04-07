import logging
import shutil

import cv2
import numpy as np

from src.core.interfaces import BaseOCREngine

try:
    import pytesseract as _pytesseract
    _PYTESSERACT_AVAILABLE = True
except ImportError:
    _PYTESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)


class TesseractEngine(BaseOCREngine):
    """Tesseract OCR implementation of BaseOCREngine (requires tesseract binary)."""

    def __init__(self, lang: str = "eng"):
        if not shutil.which("tesseract"):
            raise RuntimeError("Tesseract binary not found. Install tesseract-ocr.")
        if not _PYTESSERACT_AVAILABLE:
            raise ImportError("pytesseract not installed. Run: pip install pytesseract")
        self.pytesseract = _pytesseract
        self.lang = lang

    def predict(self, image_path: str):
        return self.pytesseract.image_to_data(
            image_path, lang=self.lang, output_type=self.pytesseract.Output.DICT
        )

    def process_image(self, image_path: str) -> tuple[object, str, list]:
        image = cv2.imread(image_path)
        if image is None:
            return None, "Failed to read image.", None

        image_with_boxes = image.copy()
        data = self.predict(image_path)

        lines = []
        raw_data = []

        for i in range(len(data["text"])):
            if int(data["conf"][i]) > 0:
                text = data["text"][i]
                score = float(data["conf"][i]) / 100.0
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                lines.append(f"Text: {text}\nConfidence: {score:.2f}\n---")
                raw_data.append({"text": text, "confidence": score, "box": box})
                cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)

        image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        if not isinstance(image_rgb, np.ndarray):
            image_rgb = None
        formatted_text = "\n".join(lines) if lines else "No text detected."
        return image_rgb, formatted_text, raw_data
