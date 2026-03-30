import cv2
import numpy as np
from paddleocr import PaddleOCR

def test():
    # Create a blank white image (portrait)
    img = np.ones((800, 400, 3), dtype=np.uint8) * 255
    
    # Draw some black text
    cv2.putText(img, "TEST 123", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "HELLO", (50, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    ocr = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)
    
    print("Running predict...")
    result = ocr.predict(img)
    
    if result and isinstance(result[0], dict) and 'rec_texts' in result[0]:
        ocr_result_dict = result[0]
        print("Texts:", ocr_result_dict['rec_texts'])
        print("Scores:", ocr_result_dict['rec_scores'])
        print("Boxes:\n", ocr_result_dict['dt_polys'])

if __name__ == "__main__":
    test()
