# app.py

import gradio as gr
from paddleocr import PaddleOCR
import numpy as np

# Initialize the PaddleOCR model for English
print("Initializing PaddleOCR model...")
ocr = PaddleOCR(use_angle_cls=True, lang='en')
print("Model initialized.")

def perform_ocr(image):
    """
    Performs OCR on the given image and formats the output.
    
    Args:
        image (numpy.ndarray): The input image as a NumPy array.
    
    Returns:
        str: A formatted string of the detected text and their confidence scores.
    """
    if image is None:
        return "Please upload an image."

    result = ocr.predict(image)
    
    if result and result[0] is not None:
        lines = []
        for line_data in result[0]:
            text = line_data[1][0]
            confidence = line_data[1][1]
            lines.append(f"Text: {text}\nConfidence: {confidence:.2f}\n---")
        
        return "\n".join(lines)
    else:
        return "No text detected in the image."

# Create the Gradio Interface
iface = gr.Interface(
    fn=perform_ocr,
    inputs=gr.Image(type="numpy", label="Upload Image for OCR"),
    outputs=gr.Textbox(label="Extracted Text", lines=15, placeholder="Text will appear here..."),
    title="📝 PaddleOCR Text Extractor",
    description="A simple web app to extract text from images using Google's PaddleOCR. Upload an image and see the magic happen!",
    article="Powered by PaddleOCR and Gradio. Hosted on Hugging Face Spaces.",
    allow_flagging="never"
)

# Queue requests and then launch the app. This is the key change!
if __name__ == "__main__":
    iface.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)