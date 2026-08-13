from abc import ABC, abstractmethod


class OcrClient(ABC):
    """OCR seam: implementations return recognized text for a document and
    raise OcrExecutionException on engine failure. Swappable for a cloud
    provider without touching the extractor."""

    @abstractmethod
    def extract_text(self, file_bytes: bytes, mime_type: str) -> str:
        raise NotImplementedError
