# core/engines/tesseract_engine.py

import cv2
import numpy as np
import shutil
from core.ocr_engine import BaseOCREngine

class TesseractEngine(BaseOCREngine):
    """
    Tesseract OCR Implementation (via pytesseract).
    Requires tesseract binary to be installed on the system.
    """
    def __init__(self, lang='eng'):
        if not shutil.which("tesseract"):
            raise RuntimeError("Tesseract binary not found. Please install tesseract-ocr.")
            
        try:
            import pytesseract
            self.pytesseract = pytesseract
            self.lang = lang
        except ImportError:
            raise ImportError("pytesseract not installed. Please install it using 'pip install pytesseract'.")

    def predict(self, image_path):
        """
        Performs OCR using Pytesseract.
        """
        # Returns a pandas-like data structure if used with output_type
        return self.pytesseract.image_to_data(image_path, lang=self.lang, output_type=self.pytesseract.Output.DICT)

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None, "Failed to read image.", None

        image_with_boxes = image.copy()
        data = self.predict(image_path)
        
        lines = []
        raw_data = []

        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0: # Filter out empty/low confidence blocks
                text = data['text'][i]
                score = float(data['conf'][i]) / 100.0
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                
                lines.append(f"Text: {text}\nConfidence: {score:.2f}\n---")
                # Box format: [[x,y], [x+w,y], [x+w,y+h], [x,y+h]] to match Paddle/Easy
                box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                
                raw_data.append({
                    "text": text,
                    "confidence": score,
                    "box": box
                })
                
                # Draw the bounding box
                cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)

        image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        formatted_text = "\n".join(lines) if lines else "No text detected."
        
        return image_with_boxes_rgb, formatted_text, raw_data
