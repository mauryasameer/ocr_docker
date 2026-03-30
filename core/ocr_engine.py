# core/ocr_engine.py

import cv2
import numpy as np
from paddleocr import PaddleOCR

class OCREngine:
    def __init__(self, lang='en', use_angle_cls=True, enable_mkldnn=False):
        """
        Initializes the PaddleOCR model.
        
        Args:
            lang (str): Language for OCR (default 'en').
            use_angle_cls (bool): Use angle classification.
            enable_mkldnn (bool): Enable MKLDNN acceleration.
        """
        print(f"Initializing PaddleOCR model (lang={lang})...")
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, enable_mkldnn=enable_mkldnn)
        print("Model initialized.")

    def predict(self, image_path):
        """
        Performs OCR on an image.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            list: Raw OCR results from PaddleOCR.
        """
        return self.ocr.predict(image_path)

    def process_image(self, image_path):
        """
        Performs OCR and processes the image to add bounding boxes.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            tuple: (annotated_image_rgb, formatted_text, raw_data)
        """
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
