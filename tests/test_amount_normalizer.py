from decimal import Decimal

import pytest

from app.utils.amount_normalizer import AmountNormalizer


@pytest.fixture
def normalizer() -> AmountNormalizer:
    return AmountNormalizer()


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("18,750", Decimal("18750")),
        ("18750", Decimal("18750")),
        ("18,750.00", Decimal("18750.00")),
        ("₹18,750.00", Decimal("18750.00")),
        ("Rs. 1,00,000/-", Decimal("100000")),
        ("INR 1,234.56", Decimal("1234.56")),
        ("  9,999  ", Decimal("9999")),
    ],
)
def test_normalizes_indian_amounts(normalizer, raw, expected):
    assert normalizer.normalize(raw) == expected


def test_formatting_variants_compare_equal(normalizer):
    # Section 6: exact equality after normalization.
    assert normalizer.normalize("18,750.00") == normalizer.normalize("18750")


@pytest.mark.parametrize("raw", ["", "   ", "abc", "Rs.", "-", "/-"])
def test_rejects_unparseable_values(normalizer, raw):
    with pytest.raises(ValueError):
        normalizer.normalize(raw)


def test_rejects_none(normalizer):
    with pytest.raises(ValueError):
        normalizer.normalize(None)
