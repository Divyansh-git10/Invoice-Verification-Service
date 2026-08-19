"""The service surfaces audit metadata (confident/llm_used/ocr_method/
llm_confidence) on ValidationResult without changing decision behavior."""
from decimal import Decimal

from app.extractors.invoice_amount_extractor import InvoiceAmountExtractor
from app.models.resolved_amount import ResolvedAmount
from app.services.invoice_verification_service import InvoiceVerificationService
from app.utils.amount_normalizer import AmountNormalizer
from app.validators.amount_validator import AmountValidator
from tests.fakes import FakeAmountResolver, FakeOcrClient

PDF = "application/pdf"


def _svc(ocr_text, resolver):
    ext = InvoiceAmountExtractor(FakeOcrClient(text=ocr_text), AmountNormalizer())
    return InvoiceVerificationService(ext, AmountValidator(), resolver=resolver)


def test_confident_path_metadata_no_llm():
    # Single clean Grand Total -> confident -> resolver bypassed.
    svc = _svc("Grand Total Rs. 18,750.00", resolver=FakeAmountResolver(result=None))
    res = svc.verify(b"%PDF", PDF, Decimal("18750.00"))

    assert res.matched is True
    assert res.confident is True
    assert res.llm_used is False
    assert res.ocr_method == "tesseract"
    assert res.llm_confidence is None


def test_escalated_override_metadata():
    # Two competing Grand Totals -> ambiguous -> LLM consulted and overrides.
    resolver = FakeAmountResolver(
        result=ResolvedAmount(amount=Decimal("1200.00"), confidence=0.9, evidence="Grand Total 1,200.00")
    )
    svc = _svc("Grand Total 1,000.00\nGrand Total 1,200.00", resolver=resolver)
    res = svc.verify(b"%PDF", PDF, Decimal("1200.00"))

    assert res.actual_amount == Decimal("1200.00")
    assert res.confident is False
    assert res.llm_used is True
    assert res.llm_confidence == 0.9
    assert res.ocr_method == "tesseract"


def test_deterministic_only_path_metadata():
    # No resolver configured -> deterministic-only branch.
    from app.models.extracted_amount import ExtractedAmount

    class _Ext:
        def extract(self, b, m):
            return ExtractedAmount(amount=Decimal("500.00"))

    svc = InvoiceVerificationService(_Ext(), AmountValidator(), resolver=None)
    res = svc.verify(b"%PDF", PDF, Decimal("500.00"))

    assert res.matched is True
    assert res.llm_used is False
    assert res.ocr_method == "tesseract"
    assert res.confident is None
