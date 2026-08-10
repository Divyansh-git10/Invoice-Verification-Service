from decimal import Decimal

import pytest

from app.models.validation_result import ValidationResult
from app.validators.amount_validator import AmountValidator


@pytest.fixture
def validator() -> AmountValidator:
    return AmountValidator()


def test_equal_amounts_match(validator):
    result = validator.validate(Decimal("18750.00"), Decimal("18750.00"))

    assert isinstance(result, ValidationResult)
    assert result.matched is True
    assert result.expected_amount == Decimal("18750.00")
    assert result.actual_amount == Decimal("18750.00")


def test_different_amounts_do_not_match(validator):
    result = validator.validate(Decimal("18750.00"), Decimal("12750.00"))

    assert result.matched is False
    assert result.expected_amount == Decimal("18750.00")
    assert result.actual_amount == Decimal("12750.00")


@pytest.mark.parametrize(
    "expected, actual",
    [
        (Decimal("18750"), Decimal("18750.00")),
        (Decimal("18750.00"), Decimal("18750")),
        (Decimal("100000"), Decimal("100000.00")),
        (Decimal("0"), Decimal("0.00")),
    ],
)
def test_integer_and_decimal_equivalents_match(validator, expected, actual):
    # Exact equality after normalization: differently-scaled but numerically
    # equal Decimal values are treated as equal.
    result = validator.validate(expected, actual)

    assert result.matched is True


@pytest.mark.parametrize(
    "expected, actual",
    [
        (Decimal("18750.00"), Decimal("18750.01")),
        (Decimal("18750.00"), Decimal("1875.00")),
        (Decimal("100000.00"), Decimal("10000.00")),
    ],
)
def test_close_but_unequal_amounts_do_not_match(validator, expected, actual):
    # No tolerance: even a one-paisa difference is a mismatch.
    result = validator.validate(expected, actual)

    assert result.matched is False


def test_realistic_fraud_scenario_mismatch(validator):
    # User claims 12750.00 but the invoice actually reads 18750.00.
    result = validator.validate(Decimal("12750.00"), Decimal("18750.00"))

    assert result.matched is False
    assert result.expected_amount == Decimal("12750.00")
    assert result.actual_amount == Decimal("18750.00")


def test_result_echoes_inputs_unchanged(validator):
    expected, actual = Decimal("4999.95"), Decimal("4999.95")
    result = validator.validate(expected, actual)

    assert result.expected_amount == expected
    assert result.actual_amount == actual
    assert result.matched is True
