import sys
from pathlib import Path
import mimetypes

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.extractors.invoice_amount_extractor import build_default_extractor
from app.core.exceptions import AmountNotFoundException


INVOICES = [
    ("1.jpg", "105200.00"),
    ("2.jpg", "27904.40"),
    ("3.jpg", "4490.00"),
    ("4.jpg", "3988.40"),
    ("5.jpg", "30180.00"),
    ("6.jpg", "143370.00"),
    ("7.jpg", "968.00"),
    ("8.jpg", "47925.00"),
    ("9.jpg", "38026.00"),
    ("10.jpg", None),
    ("11.jpg", "4490.00"),
    ("12.jpg", "68230.50"),
    ("13.jpg", "1525.00"),
]


BASE = Path(__file__).resolve().parent
extractor = build_default_extractor()

passed = 0
failed = 0

print("\n================ Real Invoice Baseline ================\n")

for filename, expected in INVOICES:
    path = BASE / filename

    try:
        mime_type, _ = mimetypes.guess_type(path.name)

        with open(path, "rb") as f:
            file_bytes = f.read()

        result = extractor.extract(
            file_bytes=file_bytes,
            mime_type=mime_type,
        )

        actual = str(result.amount)

        if expected is not None and actual == expected:
            print(f"PASS  {filename:<10} expected={expected:<12} actual={actual}")
            passed += 1
        elif expected is None:
            print(f"FAIL  {filename:<10} expected=AmountNotFound actual={actual}")
            failed += 1
        else:
            print(f"FAIL  {filename:<10} expected={expected:<12} actual={actual}")
            failed += 1

    except AmountNotFoundException:
        if expected is None:
            print(f"PASS  {filename:<10} expected=AmountNotFound actual=AmountNotFound")
            passed += 1
        else:
            print(f"FAIL  {filename:<10} expected={expected:<12} actual=AmountNotFound")
            failed += 1

    except Exception as exc:
        print(f"ERROR {filename:<10} {type(exc).__name__}: {exc}")
        failed += 1


print("\n=========================================================")
print(f"Passed : {passed}")
print(f"Failed : {failed}")
print(f"Total  : {len(INVOICES)}")
print("=========================================================\n")