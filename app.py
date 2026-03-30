# app.py

import gradio as gr
from core.ocr_engine import OCRFactory

def perform_ocr(image_path, engine_name):
    """
    Performs OCR using the selected engine and returns the annotated image and text.
    """
    if image_path is None:
        return None, "Please upload an image.", "Waiting for engine..."

    try:
        # Get the selected engine from the factory
        engine = OCRFactory.get_engine(engine_name)
        annotated_image, formatted_text, _ = engine.process_image(image_path)
        import numpy as np
        from PIL import Image
        import os
        if isinstance(annotated_image, np.ndarray):
            # Reject dtype=object arrays (e.g. np.array wrapping a dict) —
            # they pass isinstance but crash Gradio's image encoder.
            if annotated_image.ndim != 3 or annotated_image.dtype == object:
                if os.environ.get("OCR_DEBUG", "0") == "1":
                    print(f"[DEBUG] Bad ndarray shape/dtype: shape={annotated_image.shape} dtype={annotated_image.dtype}")
                annotated_image = None
        elif not isinstance(annotated_image, Image.Image):
            if os.environ.get("OCR_DEBUG", "0") == "1":
                print(f"[DEBUG] Invalid image type returned: {type(annotated_image)} | Value: {repr(annotated_image)}")
            annotated_image = None
        return annotated_image, formatted_text, f"Success (using {engine_name})"
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return None, error_msg, f"Failed ({engine_name})"

# Define the Gradio interface
with gr.Blocks(title="🛡️ OCR Framework Audit Suite") as demo:
    gr.Markdown("# 🛡️ OCR Framework Audit Suite")
    gr.Markdown("Select an engine and upload an image to perform text extraction with bounding boxes.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Upload Image for OCR")
            engine_select = gr.Dropdown(
                choices=OCRFactory.list_available_engines(),
                value="paddle",
                label="Select OCR Engine"
            )
            status_output = gr.Textbox(label="Status")
            submit_btn = gr.Button("Extract Text", variant="primary")
            
        with gr.Column(scale=2):
            image_output = gr.Image(label="Image with Bounding Boxes")
            text_output = gr.Textbox(label="Extracted Text", lines=20)

    submit_btn.click(
        fn=perform_ocr,
        inputs=[image_input, engine_select],
        outputs=[image_output, text_output, status_output]
    )
    
    gr.Markdown("---")
    gr.Markdown("Powered by PaddleOCR, EasyOCR, and Tesseract. Modular architecture inspired by professional evaluation frameworks.")

if __name__ == "__main__":
    # Enable queue to prevent DDOS/crashes (Required for HF Spaces)
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)