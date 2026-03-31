# core/ocr_engine.py

import cv2
import numpy as np
from abc import ABC, abstractmethod

class BaseOCREngine(ABC):
    """
    Abstract Base Class for OCR Engines.
    All engines must implement predict and process_image.
    """
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, image_path):
        """Performs OCR and returns raw results."""
        pass

    @abstractmethod
    def process_image(self, image_path):
        """Processes image and returns (annotated_image, formatted_text, raw_data)."""
        pass

class PaddleOCREngine(BaseOCREngine):
    """
    PaddleOCR Implementation.
    """
    def __init__(self, lang='en', use_angle_cls=True, enable_mkldnn=False):
        from paddleocr import PaddleOCR
        print(f"Initializing PaddleOCR (lang={lang})...")
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, enable_mkldnn=enable_mkldnn)

    def predict(self, image_path):
        return self.ocr.predict(image_path)

    def process_image(self, image_path):
        import tempfile, os
        image = cv2.imread(image_path)
        if image is None:
            return None, "Failed to read image.", None

        result = self.ocr.predict(image_path)
        if not result:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "No text detected.", []

        lines = []
        raw_data = []
        result_item = result[0]

        if isinstance(result_item, dict) and 'rec_texts' in result_item:
            texts  = result_item.get('rec_texts', [])
            scores = result_item.get('rec_scores', [])
            boxes  = result_item.get('dt_polys', [])
            for text, score, box in zip(texts, scores, boxes):
                lines.append(f"Text: {text}\nConfidence: {score:.2f}\n---")
                pts = np.array(box, dtype=np.int32)
                raw_data.append({
                    "text": text,
                    "confidence": float(score),
                    "box": pts.tolist()
                })

        # Use PaddleX's own visualization — it applies the exact same transforms
        # used during inference, so bounding boxes are guaranteed to be aligned.
        annotated_bgr = None
        if hasattr(result_item, 'save_to_img'):
            try:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name
                result_item.save_to_img(tmp_path)
                annotated_bgr = cv2.imread(tmp_path)
                os.unlink(tmp_path)
            except Exception as e:
                print(f"[WARN] save_to_img failed: {e}")

        if annotated_bgr is None or annotated_bgr.ndim != 3:
            # Fallback: draw manually on original image
            annotated_bgr = image.copy()
            for entry in raw_data:
                pts = np.array(entry['box'], dtype=np.int32)
                cv2.polylines(annotated_bgr, [pts], isClosed=True,
                              color=(0, 255, 0), thickness=2)

        image_with_boxes_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        formatted_text = "\n".join(lines) if lines else "No text detected."
        return image_with_boxes_rgb, formatted_text, raw_data

class OCRFactory:
    """
    Factory to manage and instantiate OCR engines.
    """
    _engines = {
        "paddle": PaddleOCREngine
    }

    @classmethod
    def register_engine(cls, name, engine_class):
        cls._engines[name.lower()] = engine_class

    @classmethod
    def get_engine(cls, name="paddle", **kwargs):
        name = name.lower()
        if name not in cls._engines:
            # Attempt to dynamic load
            if name == "easyocr":
                from core.engines.easyocr_engine import EasyOCREngine
                cls.register_engine("easyocr", EasyOCREngine)
            elif name == "tesseract":
                # We will implement this next
                from core.engines.tesseract_engine import TesseractEngine
                cls.register_engine("tesseract", TesseractEngine)
                
        engine_class = cls._engines.get(name)
        if not engine_class:
            raise ValueError(f"OCR Engine '{name}' not found or not installed.")
        return engine_class(**kwargs)

    @classmethod
    def list_available_engines(cls):
        # In a real framework, we'd scan core/engines/
        return ["paddle", "easyocr", "tesseract"]
