# app.py

import gradio as gr
from core.ocr_engine import OCREngine

# Initialize the modular OCR engine
engine = OCREngine(lang='en', use_angle_cls=True, enable_mkldnn=False)

def perform_ocr(image_path):
    """
    Performs OCR using the modular engine and returns the annotated image and text.
    """
    if image_path is None:
        return None, "Please upload an image."

    annotated_image, formatted_text, _ = engine.process_image(image_path)
    return annotated_image, formatted_text

# Define the Gradio interface
iface = gr.Interface(
    fn=perform_ocr,
    inputs=gr.Image(type="filepath", label="Upload Image for OCR"),
    outputs=[
        gr.Image(label="Image with Bounding Boxes"),
        gr.Textbox(label="Extracted Text", lines=20)
    ],
    title="📝 PaddleOCR Text Extractor with Bounding Boxes",
    description="Upload an image to see the detected text highlighted with bounding boxes and listed with confidence scores.",
    article="Powered by PaddleOCR, OpenCV, and Gradio. Modular architecture inspired by professional evaluation frameworks.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)