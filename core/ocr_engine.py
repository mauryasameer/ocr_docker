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

        # Pass file path so PaddleOCR handles its own loading.
        result = self.ocr.predict(image_path)
        if not result:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "No text detected.", []

        lines = []
        raw_data = []
        result_item = result[0]

        # PaddleOCR may resize the image internally before inference.
        # dt_polys are in that resized space, so compute scale factors to map
        # coordinates back onto the original full-resolution cv2.imread image.
        scale_x, scale_y = 1.0, 1.0
        if (hasattr(result_item, 'img')
                and isinstance(result_item.img, np.ndarray)
                and result_item.img.ndim == 3):
            ocr_h, ocr_w = result_item.img.shape[:2]
            orig_h, orig_w = image.shape[:2]
            if ocr_h > 0 and ocr_w > 0:
                scale_x = orig_w / ocr_w
                scale_y = orig_h / ocr_h

        image_with_boxes = image.copy()

        if isinstance(result_item, dict) and 'rec_texts' in result_item:
            texts  = result_item.get('rec_texts', [])
            scores = result_item.get('rec_scores', [])
            boxes  = result_item.get('dt_polys', [])

            for text, score, box in zip(texts, scores, boxes):
                lines.append(f"Text: {text}\nConfidence: {score:.2f}\n---")

                pts = np.array(box, dtype=np.float32)
                pts[:, 0] *= scale_x
                pts[:, 1] *= scale_y
                pts = pts.astype(np.int32)

                raw_data.append({
                    "text": text,
                    "confidence": float(score),
                    "box": pts.tolist()
                })

                cv2.polylines(image_with_boxes, [pts], isClosed=True,
                              color=(0, 255, 0), thickness=2)

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
