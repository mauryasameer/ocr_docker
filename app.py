# app.py

import gradio as gr
from paddleocr import PaddleOCR
import numpy as np
import cv2  # Import OpenCV for drawing

# Initialize the PaddleOCR model for English
print("Initializing PaddleOCR model...")
ocr = PaddleOCR(use_angle_cls=True, lang='en')
print("Model initialized.")

def perform_ocr(image):
    """
    Performs OCR, draws bounding boxes, and returns the annotated image and text.
    
    Args:
        image (numpy.ndarray): The input image as a NumPy array (expected RGB).
    
    Returns:
        tuple: A tuple containing (annotated_image, formatted_text).
    """
    if image is None:
        return None, "Please upload an image."

    # --- Start of updated logic ---
    # Ensure image is in 8-bit unsigned integer format (0-255)
    # and in RGB color format, which Gradio's numpy type usually provides.
    # If Gradio passes float values (0.0-1.0), this converts them to 0-255.
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # PaddleOCR's predict method expects a NumPy array (H, W, C) which is typically RGB.
    # Create a copy of the image to draw on, leaving the original untouched
    image_with_boxes = image.copy() 
    
    result = ocr.predict(image) # Pass the (hopefully) consistent image to PaddleOCR
    
    lines = []
    # Check if the result is a list and contains a dictionary
    if result and isinstance(result[0], dict) and 'rec_texts' in result[0]:
        
        ocr_result_dict = result[0]
        texts = ocr_result_dict['rec_texts']
        scores = ocr_result_dict['rec_scores']
        boxes = ocr_result_dict['dt_polys'] # Get the bounding box polygons

        # Combine texts, scores, and boxes
        for text, score, box in zip(texts, scores, boxes):
            # Format text for output
            lines.append(f"Text: {text}\nConfidence: {score:.2f}\n---")
            
            # Draw the bounding box on the image
            # Convert the box points to an integer NumPy array
            # The 'box' variable from PaddleOCR's dt_polys is typically already [x,y] float coordinates for the polygon.
            # Converting to int32 is crucial for cv2.polylines.
            pts = np.array(box, dtype=np.int32)
            
            # Ensure the points are within image bounds before drawing (defensive check)
            # This is more for preventing errors than fixing offset, but good practice.
            # pts[:, 0] = np.clip(pts[:, 0], 0, image.shape[1] - 1) # x coordinates
            # pts[:, 1] = np.clip(pts[:, 1], 0, image.shape[0] - 1) # y coordinates
            
            # Use cv2.polylines to draw the polygon. Color is (B, G, R) for OpenCV,
            # but since image_with_boxes is RGB from Gradio, (0, 255, 0) is green.
            cv2.polylines(image_with_boxes, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    formatted_text = "\n".join(lines) if lines else "No text detected in the image."
    
    return image_with_boxes, formatted_text
    # --- End of updated logic ---
    
# --- Update the Interface with two outputs ---
iface = gr.Interface(
    fn=perform_ocr,
    inputs=gr.Image(type="numpy", label="Upload Image for OCR"),
    outputs=[
        gr.Image(label="Image with Bounding Boxes"),
        gr.Textbox(label="Extracted Text", lines=20)
    ],
    title="📝 PaddleOCR Text Extractor with Bounding Boxes",
    description="Upload an image to see the detected text highlighted with bounding boxes and listed with confidence scores.",
    article="Powered by PaddleOCR, OpenCV, and Gradio. Hosted on Hugging Face Spaces.",
    allow_flagging="never"
)

# Queue requests and then launch the app
if __name__ == "__main__":
    iface.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)