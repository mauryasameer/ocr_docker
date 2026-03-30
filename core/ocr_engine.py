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
        image = cv2.imread(image_path)
        if image is None:
            return None, "Failed to read image.", None

        image_with_boxes = image.copy()
        result = self.predict(image_path)
        
        lines = []
        raw_data = []

        if result and isinstance(result[0], dict) and 'rec_texts' in result[0]:
            ocr_result_dict = result[0]
            texts = ocr_result_dict['rec_texts']
            scores = ocr_result_dict['rec_scores']
            boxes = ocr_result_dict['dt_polys']

            for text, score, box in zip(texts, scores, boxes):
                lines.append(f"Text: {text}\nConfidence: {score:.2f}\n---")
                raw_data.append({
                    "text": text,
                    "confidence": float(score),
                    "box": box.tolist() if isinstance(box, np.ndarray) else box
                })
                
                pts = np.array(box, dtype=np.int32)
                cv2.polylines(image_with_boxes, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
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
