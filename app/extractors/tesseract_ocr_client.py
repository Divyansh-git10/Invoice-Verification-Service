import io

from app.core.exceptions import OcrExecutionException
from app.core.logger import get_logger
from app.extractors.ocr_client import OcrClient

logger = get_logger(__name__)


class TesseractOcrClient(OcrClient):
    """OCR backed by local Tesseract. Requires the `tesseract` binary plus
    pytesseract/Pillow/PyMuPDF. PDFs use their embedded text layer when
    present, otherwise each page is rasterized and OCR'd. Heavy deps are
    imported lazily so fake-OCR unit tests don't need them installed."""

    def __init__(self, tesseract_lang: str = "eng", render_dpi: int = 300):
        self._lang = tesseract_lang
        self._render_dpi = render_dpi

    def extract_text(self, file_bytes: bytes, mime_type: str) -> str:
        try:
            if mime_type == "application/pdf":
                text = self._ocr_pdf(file_bytes)
            else:
                text = self._ocr_image(file_bytes)
        except OcrExecutionException:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize any engine error
            logger.exception("Tesseract OCR failed")
            raise OcrExecutionException(f"Tesseract OCR failed: {exc}") from exc

        logger.info("OCR recognized %d characters", len(text or ""))
        return text or ""

    def _ocr_image(self, file_bytes: bytes) -> str:
        import pytesseract
        from PIL import Image

        with Image.open(io.BytesIO(file_bytes)) as image:
            return pytesseract.image_to_string(image, lang=self._lang)

    def _ocr_pdf(self, file_bytes: bytes) -> str:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image

        pages: list[str] = []
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            for page in document:
                embedded = page.get_text().strip()
                if embedded:
                    pages.append(embedded)
                    continue

                pixmap = page.get_pixmap(dpi=self._render_dpi)
                with Image.open(io.BytesIO(pixmap.tobytes("png"))) as image:
                    pages.append(pytesseract.image_to_string(image, lang=self._lang))

        return "\n".join(pages)
