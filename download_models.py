from paddleocr import PaddleOCR

# This initialization will trigger the download of the default English models
# and store them in the default cache directory inside the Docker image.
print("Starting model pre-download...")
PaddleOCR(use_angle_cls=True, lang='en')
print("Models pre-downloaded successfully.")