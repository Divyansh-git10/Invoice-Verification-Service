from abc import ABC, abstractmethod


class OcrClient(ABC):
    """Injection seam for optical character recognition.

    This is the stable contract every extractor depends on. V1 ships a
    local Tesseract implementation (`TesseractOcrClient`). A cloud provider
    (e.g. Azure AI Document Intelligence) can be introduced later as another
    `OcrClient` implementation and injected into the extractor without any
    change to the extraction pipeline or the surrounding architecture.

    Implementations must be stateless with respect to a single call: given
    the raw bytes of a document and its MIME type, return the recognized
    text. Recognition failures must be raised as `OcrExecutionException`.
    """

    @abstractmethod
    def extract_text(self, file_bytes: bytes, mime_type: str) -> str:
        """Return the text recognized in the document.

        Args:
            file_bytes: Raw bytes of the uploaded document.
            mime_type: MIME type of the document (e.g. "application/pdf").

        Returns:
            The recognized text. May be an empty string if the document
            contains no readable text.

        Raises:
            OcrExecutionException: If the OCR engine fails to run.
        """
        raise NotImplementedError
