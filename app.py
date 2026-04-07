import logging
import os
import traceback

import gradio as gr
import numpy as np
from PIL import Image

from src.providers.paddle_provider import OCRFactory

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def perform_ocr(image_path: str, engine_name: str) -> tuple:
    if image_path is None:
        return None, "Please upload an image.", "Waiting for engine..."

    try:
        engine = OCRFactory.get_engine(engine_name)
        annotated_image, formatted_text, _ = engine.process_image(image_path)

        if isinstance(annotated_image, np.ndarray):
            if annotated_image.ndim != 3 or annotated_image.dtype == object:
                if os.environ.get("OCR_DEBUG", "0") == "1":
                    logger.debug(
                        "Bad ndarray shape/dtype: shape=%s dtype=%s",
                        annotated_image.shape, annotated_image.dtype,
                    )
                annotated_image = None
        elif not isinstance(annotated_image, Image.Image):
            if os.environ.get("OCR_DEBUG", "0") == "1":
                logger.debug("Invalid image type returned: %s", type(annotated_image))
            annotated_image = None

        return annotated_image, formatted_text, f"Success (using {engine_name})"
    except Exception as e:
        error_msg = f"Error: {e}"
        logger.error("OCR failed for engine %s", engine_name, exc_info=True)
        traceback.print_exc()
        return None, error_msg, f"Failed ({engine_name})"


with gr.Blocks(title="OCR Framework Audit Suite") as demo:
    gr.Markdown("# OCR Framework Audit Suite")
    gr.Markdown("Select an engine and upload an image to perform text extraction with bounding boxes.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Upload Image for OCR")
            engine_select = gr.Dropdown(
                choices=OCRFactory.list_available_engines(),
                value="paddle",
                label="Select OCR Engine",
            )
            status_output = gr.Textbox(label="Status")
            submit_btn = gr.Button("Extract Text", variant="primary")

        with gr.Column(scale=2):
            image_output = gr.Image(label="Image with Bounding Boxes")
            text_output = gr.Textbox(label="Extracted Text", lines=20)

    submit_btn.click(
        fn=perform_ocr,
        inputs=[image_input, engine_select],
        outputs=[image_output, text_output, status_output],
    )

    gr.Markdown("---")
    gr.Markdown("Powered by PaddleOCR, EasyOCR, and Tesseract.")

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
