import sys
from pathlib import Path
import mimetypes

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.extractors.tesseract_ocr_client import TesseractOcrClient


if len(sys.argv) != 2:
    print("Usage: python tests/debug_ocr.py <invoice_path>")
    sys.exit(1)

invoice_path = Path(sys.argv[1])

if not invoice_path.exists():
    print(f"File not found: {invoice_path}")
    sys.exit(1)

mime_type, _ = mimetypes.guess_type(invoice_path)

if mime_type is None:
    print(f"Could not determine MIME type for: {invoice_path}")
    sys.exit(1)

with open(invoice_path, "rb") as f:
    file_bytes = f.read()

ocr = TesseractOcrClient()

text = ocr.extract_text(
    file_bytes=file_bytes,
    mime_type=mime_type,
)

print("=" * 80)
print(f"FILE: {invoice_path}")
print(f"MIME: {mime_type}")
print("=" * 80)
print(text)
print("=" * 80)