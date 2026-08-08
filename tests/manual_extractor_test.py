import json
import mimetypes
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.exceptions import AmountNotFoundException
from app.extractors.invoice_amount_extractor import build_default_extractor

# -------------------------------------------------------

MANIFEST_PATH = Path("tests/fixtures/manifest.json")

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)

extractor = build_default_extractor()

print("\n================ Invoice Extraction Regression ================\n")

passed = 0
failed = 0

for invoice in manifest["invoices"]:

    invoice_path = Path("tests/fixtures") / invoice["file"]

    expected = invoice["expected_total"]

    mime_type, _ = mimetypes.guess_type(invoice_path)

    try:

        with open(invoice_path, "rb") as f:
            file_bytes = f.read()

        result = extractor.extract(
            file_bytes=file_bytes,
            mime_type=mime_type,
        )

        actual = Decimal(str(result.amount))

        # Expected exception but extractor succeeded
        if expected is None:

            print(
                f"❌ {invoice['id']:<12} "
                f"Expected AmountNotFound "
                f"Got {actual}"
            )
            failed += 1
            continue

        expected_decimal = Decimal(expected)

        if actual == expected_decimal:

            print(
                f"✅ {invoice['id']:<12} "
                f"{actual}"
            )
            passed += 1

        else:

            print(
                f"❌ {invoice['id']:<12} "
                f"Expected {expected_decimal} "
                f"Got {actual}"
            )
            failed += 1

    except AmountNotFoundException:

        if expected is None:

            print(
                f"✅ {invoice['id']:<12} "
                f"AmountNotFound"
            )
            passed += 1

        else:

            print(
                f"❌ {invoice['id']:<12} "
                f"Expected {expected} "
                f"Got AmountNotFound"
            )
            failed += 1

    except Exception as e:

        print(
            f"❌ {invoice['id']:<12} "
            f"{type(e).__name__}: {e}"
        )
        failed += 1

print("\n==============================================================")
print(f"Passed : {passed}")
print(f"Failed : {failed}")
print("==============================================================")