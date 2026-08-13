import re
from decimal import Decimal, InvalidOperation


class AmountNormalizer:
    """Normalizes raw amount strings to a canonical Decimal. INR only:
    commas (incl. Indian grouping like 1,00,000) are stripped, the dot is the
    decimal separator, and currency noise (₹/Rs/INR, trailing "/-") is removed.
    e.g. "Rs. 1,00,000/-" -> Decimal("100000")."""

    _NOISE = re.compile(r"[^0-9.,]")
    # Guards against a stray dot (e.g. the "." in a "Rs." prefix) leaking in.
    _NUMBER = re.compile(r"\d+(?:\.\d+)?")

    def normalize(self, raw: str) -> Decimal:
        """Return the canonical Decimal, or raise ValueError if unparseable."""
        if raw is None:
            raise ValueError("Cannot normalize a null amount")

        cleaned = self._NOISE.sub("", str(raw))
        # Remove thousands separators; keep the decimal point.
        cleaned = cleaned.replace(",", "")

        match = self._NUMBER.search(cleaned)
        if match is None:
            raise ValueError(f"No numeric value found in: {raw!r}")

        try:
            return Decimal(match.group())
        except InvalidOperation as exc:
            raise ValueError(f"Cannot parse amount: {raw!r}") from exc
