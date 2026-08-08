import sys
from pathlib import Path
import mimetypes

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.extractors.tesseract_ocr_client import TesseractOcrClient

invoice_path = Path("tests/fixtures/invoices/invoice_11.png")

mime_type, _ = mimetypes.guess_type(invoice_path)

with open(invoice_path, "rb") as f:
    file_bytes = f.read()

ocr = TesseractOcrClient()

text = ocr.extract_text(
    file_bytes=file_bytes,
    mime_type=mime_type,
)

print("=" * 80)
print(text)
print("=" * 80)
