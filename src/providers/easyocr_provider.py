from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from src.core.interfaces import BaseOCREngine

try:
    import easyocr as _easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False

logger = logging.getLogger(__name__)


class EasyOCREngine(BaseOCREngine):
    """EasyOCR implementation of BaseOCREngine."""

    def __init__(self, gpu: bool = False, lang: Optional[list[str]] = None):
        if not _EASYOCR_AVAILABLE:
            raise ImportError("easyocr not installed. Run: pip install easyocr")
        if lang is None:
            lang = ["en"]
        logger.info("Initializing EasyOCR (gpu=%s, lang=%s)...", gpu, lang)
        self.reader = _easyocr.Reader(lang, gpu=gpu)

    def predict(self, image_path: str):
        return self.reader.readtext(image_path)

    def process_image(self, image_path: str) -> tuple[object, str, list]:
        image = cv2.imread(image_path)
        if image is None:
            return None, "Failed to read image.", None

        image_with_boxes = image.copy()
        result = self.predict(image_path)

        lines = []
        raw_data = []

        for box, text, score in result:
            lines.append(f"Text: {text}\nConfidence: {score:.2f}\n---")
            raw_data.append({"text": text, "confidence": float(score), "box": box})
            pts = np.array(box, dtype=np.int32)
            cv2.polylines(image_with_boxes, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        if not isinstance(image_rgb, np.ndarray):
            image_rgb = None
        formatted_text = "\n".join(lines) if lines else "No text detected."
        return image_rgb, formatted_text, raw_data
