import re
from decimal import Decimal, InvalidOperation


class AmountNormalizer:
    """Normalizes raw amount strings into a canonical `Decimal`.

    V1 scope is Indian (INR) formatting only, per the architecture's
    Locale & Normalization Assumptions:

    - Thousands separators are commas, including Indian grouping
      (e.g. "1,00,000") and are removed.
    - The decimal separator is a dot and is preserved.
    - Currency noise (₹, "Rs", "Rs.", "INR") and trailing marks such as
      "/-" are stripped.

    Examples (all treated as equivalent numeric values):
        "18,750"     -> Decimal("18750")
        "18750"      -> Decimal("18750")
        "18,750.00"  -> Decimal("18750.00")
        "Rs. 1,00,000/-" -> Decimal("100000")

    This class performs no business logic and no rounding beyond parsing.
    """

    # Anything that is not a digit, comma, or dot is treated as noise.
    _NOISE = re.compile(r"[^0-9.,]")
    # A numeric value once separators are removed (guards against stray dots,
    # e.g. the "." in a "Rs." prefix, leaking into the result).
    _NUMBER = re.compile(r"\d+(?:\.\d+)?")

    def normalize(self, raw: str) -> Decimal:
        """Return the canonical `Decimal` for a raw amount string.

        Raises:
            ValueError: If `raw` contains no parseable numeric value.
        """
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
