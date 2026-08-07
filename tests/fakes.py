from app.extractors.ocr_client import OcrClient


class FakeOcrClient(OcrClient):
    """Test double for the OCR seam.

    Returns canned text (or raises a canned error) so the extractor's
    parsing/identification logic can be tested without Tesseract.
    """

    def __init__(self, text: str = "", raises: Exception | None = None):
        self._text = text
        self._raises = raises
        self.calls: list[tuple[int, str]] = []

    def extract_text(self, file_bytes: bytes, mime_type: str) -> str:
        self.calls.append((len(file_bytes), mime_type))
        if self._raises is not None:
            raise self._raises
        return self._text
