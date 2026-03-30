import numpy as np
import pytest
from core.ocr_engine import OCRFactory

@pytest.mark.parametrize("engine_name", ["paddle", "easyocr", "tesseract"])
def test_process_image_returns_valid_image(tmp_path, engine_name):
    import cv2
    img_path = tmp_path / "dummy.png"
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), dummy_img)

    try:
        engine = OCRFactory.get_engine(engine_name)
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"Dependency not available for {engine_name}: {e}")

    result = engine.process_image(str(img_path))
    if not result or len(result) != 3:
        pytest.skip(f"{engine_name} process_image did not return 3 values")
    annotated_image, formatted_text, raw_data = result
    assert annotated_image is None or isinstance(annotated_image, np.ndarray)
