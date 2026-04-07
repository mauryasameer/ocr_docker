from abc import ABC, abstractmethod


class BaseOCREngine(ABC):
    """Abstract base class for all OCR engine implementations."""

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, image_path: str):
        """Perform OCR and return raw engine results."""
        pass

    @abstractmethod
    def process_image(self, image_path: str) -> tuple[object, str, list]:
        """Process image and return (annotated_image, formatted_text, raw_data)."""
        pass
