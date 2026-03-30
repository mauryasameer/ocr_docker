import numpy as np
import pytest
from core.ocr_engine import OCRFactory

@pytest.mark.parametrize("engine_name", ["paddle", "easyocr", "tesseract"])
def test_process_image_returns_valid_image(tmp_path, engine_name):
    # Create a dummy image file
    img_path = tmp_path / "dummy.png"
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    import cv2
    cv2.imwrite(str(img_path), dummy_img)

    engine = OCRFactory.get_engine(engine_name)
    annotated_image, formatted_text, raw_data = engine.process_image(str(img_path))
    # The output must be a numpy array or None (if image reading fails)
    assert annotated_image is None or isinstance(annotated_image, np.ndarray)
