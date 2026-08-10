from decimal import Decimal

from app.models.validation_result import ValidationResult


class AmountValidator:
    """Business comparison between an expected amount and the extracted amount.

    Pure business rule, per the architecture:
      - depends only on the domain model (`ValidationResult`);
      - performs no OCR, no file handling, no HTTP concerns.

    Match semantics (V1): EXACT numeric equality of `Decimal` values, with no
    tolerance and no fuzzy comparison. Python's `Decimal` equality compares by
    numeric value, so differently-scaled but equal values are treated as equal
    (e.g. Decimal("18750") == Decimal("18750.00")). This is the "exact equality
    after normalization" rule: normalization is owned upstream (the extractor /
    normalizer), and this validator only compares the resulting canonical
    Decimal values.

    If tolerance is ever required by the business, it should be added here as a
    validation rule rather than by changing the domain contracts.
    """

    def validate(
        self,
        expected_amount: Decimal,
        actual_amount: Decimal,
    ) -> ValidationResult:
        """Compare the two amounts and return a `ValidationResult`.

        Args:
            expected_amount: The amount entered by the user.
            actual_amount: The amount extracted from the invoice.

        Returns:
            A `ValidationResult` echoing both amounts and whether they match.
        """
        matched = expected_amount == actual_amount

        return ValidationResult(
            matched=matched,
            expected_amount=expected_amount,
            actual_amount=actual_amount,
        )
