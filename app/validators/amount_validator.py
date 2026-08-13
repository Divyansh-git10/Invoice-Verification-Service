from decimal import Decimal

from app.models.validation_result import ValidationResult


class AmountValidator:
    """Compares expected vs extracted amount by exact Decimal equality (no
    tolerance). Decimal equality is value-based, so 18750 == 18750.00."""

    def validate(
        self,
        expected_amount: Decimal,
        actual_amount: Decimal,
    ) -> ValidationResult:
        matched = expected_amount == actual_amount

        return ValidationResult(
            matched=matched,
            expected_amount=expected_amount,
            actual_amount=actual_amount,
        )
