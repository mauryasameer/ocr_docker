# core/engines/easyocr_engine.py

import cv2
import numpy as np
from core.ocr_engine import BaseOCREngine

class EasyOCREngine(BaseOCREngine):
    """
    EasyOCR Implementation.
    """
    def __init__(self, gpu=False, lang=['en']):
        try:
            import easyocr
            print(f"Initializing EasyOCR (gpu={gpu}, lang={lang})...")
            self.reader = easyocr.Reader(lang, gpu=gpu)
        except ImportError:
            raise ImportError("easyocr not installed. Please install it using 'pip install easyocr'.")

    def predict(self, image_path):
        """
        Performs OCR using EasyOCR.
        Returns format: [([[x, y], [x, y], [x, y], [x, y]], 'text', 0.9)]
        """
        return self.reader.readtext(image_path)

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None, "Failed to read image.", None

        image_with_boxes = image.copy()
        result = self.predict(image_path)
        
        lines = []
        raw_data = []

        if result:
            for (box, text, score) in result:
                lines.append(f"Text: {text}\nConfidence: {score:.2f}\n---")
                raw_data.append({
                    "text": text,
                    "confidence": float(score),
                    "box": box # Already a list of points
                })
                
                # Draw the bounding box
                pts = np.array(box, dtype=np.int32)
                cv2.polylines(image_with_boxes, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        formatted_text = "\n".join(lines) if lines else "No text detected."
        
        return image_with_boxes_rgb, formatted_text, raw_data
