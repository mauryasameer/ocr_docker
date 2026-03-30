import cv2
import numpy as np
import os
from paddleocr import PaddleOCR

def download_and_warmup():
    print("--- Starting PaddleOCR Model Pre-download ---")
    # This triggers the default model downloads
    ocr = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)
    
    # Create a tiny 64x64 blank image for warm-up
    warmup_img_path = "warmup_tmp.png"
    cv2.imwrite(warmup_img_path, np.zeros((64, 64, 3), dtype=np.uint8))
    
    print("--- Performing Warm-up Inference (triggers font & secondary model downloads) ---")
    try:
        # This triggers internal downloads for UVDoc, fonts (simfang.ttf), etc.
        ocr.predict(warmup_img_path)
        print("--- Warm-up Successful. All models and fonts cached. ---")
    except Exception as e:
        print(f"--- Warm-up Notice: {e} ---")
    finally:
        if os.path.exists(warmup_img_path):
            os.remove(warmup_img_path)

if __name__ == "__main__":
    download_and_warmup()