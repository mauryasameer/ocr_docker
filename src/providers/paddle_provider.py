from __future__ import annotations

import logging
from typing import Tuple
import os
import tempfile

import cv2
import numpy as np

from src.core.interfaces import BaseOCREngine

# PaddleOCR is an optional heavy dependency — imported lazily so the module
# can be imported without it installed (e.g. in CI or when using other engines).
try:
    from paddleocr import PaddleOCR as _PaddleOCR
    _PADDLE_AVAILABLE = True
except ImportError:
    _PADDLE_AVAILABLE = False

logger = logging.getLogger(__name__)


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR implementation of BaseOCREngine."""

    def __init__(self, lang: str = "en", use_angle_cls: bool = True, enable_mkldnn: bool = False):
        if not _PADDLE_AVAILABLE:
            raise ImportError("paddleocr not installed. Run: pip install paddleocr")
        logger.info("Initializing PaddleOCR (lang=%s)...", lang)
        self.ocr = _PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, enable_mkldnn=enable_mkldnn)

    def predict(self, image_path: str):
        return self.ocr.predict(image_path)

    def process_image(self, image_path: str) -> Tuple[object, str, list]:
        image = cv2.imread(image_path)
        if image is None:
            return None, "Failed to read image.", None

        result = self.ocr.predict(image_path)
        if not result:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "No text detected.", []

        lines = []
        raw_data = []
        result_item = result[0]

        if isinstance(result_item, dict) and "rec_texts" in result_item:
            texts = result_item.get("rec_texts", [])
            scores = result_item.get("rec_scores", [])
            boxes = result_item.get("dt_polys", [])
            for text, score, box in zip(texts, scores, boxes):
                lines.append(f"Text: {text}\nConfidence: {score:.2f}\n---")
                pts = np.array(box, dtype=np.int32)
                raw_data.append({"text": text, "confidence": float(score), "box": pts.tolist()})

        annotated_bgr = None
        if hasattr(result_item, "save_to_img"):
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                result_item.save_to_img(tmp_path)
                annotated_bgr = cv2.imread(tmp_path)
                os.unlink(tmp_path)
            except Exception:
                logger.warning("save_to_img failed; falling back to manual bounding box draw")

        if annotated_bgr is None or annotated_bgr.ndim != 3:
            annotated_bgr = image.copy()
            for entry in raw_data:
                pts = np.array(entry["box"], dtype=np.int32)
                cv2.polylines(annotated_bgr, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        image_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        formatted_text = "\n".join(lines) if lines else "No text detected."
        return image_rgb, formatted_text, raw_data


class OCRFactory:
    """Factory for instantiating and registering OCR engine providers."""

    _engines: dict = {"paddle": PaddleOCREngine}

    @classmethod
    def register_engine(cls, name: str, engine_class: type) -> None:
        cls._engines[name.lower()] = engine_class

    @classmethod
    def get_engine(cls, name: str = "paddle", **kwargs) -> BaseOCREngine:
        name = name.lower()
        if name not in cls._engines:
            if name == "easyocr":
                from src.providers.easyocr_provider import EasyOCREngine
                cls.register_engine("easyocr", EasyOCREngine)
            elif name == "tesseract":
                from src.providers.tesseract_provider import TesseractEngine
                cls.register_engine("tesseract", TesseractEngine)
        engine_class = cls._engines.get(name)
        if not engine_class:
            raise ValueError(f"OCR Engine '{name}' not found or not installed.")
        return engine_class(**kwargs)

    @classmethod
    def list_available_engines(cls) -> list[str]:
        return ["paddle", "easyocr", "tesseract"]
