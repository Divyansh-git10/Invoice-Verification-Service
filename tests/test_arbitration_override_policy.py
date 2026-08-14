"""Contextual override policy (replaces the absolute label-priority veto).

Covers scenarios A-J from the task. No real LLM calls (provider stubbed).

Priority reference (lower = more authoritative):
  invoice amount=0, invoice total=1, grand total=2, ..., total amount=10, total=11
"""
from decimal import Decimal

import httpx

from app.extractors.invoice_amount_extractor import (
    InvoiceAmountExtractor,
    LabelledCandidate,
)
from app.services.invoice_verification_service import InvoiceVerificationService
from app.utils.amount_normalizer import AmountNormalizer
from app.validators.amount_validator import AmountValidator
from tests.fakes import FakeOcrClient
from tests.test_llm_amount_resolver import StubResolver, ctx, lc

PDF = "application/pdf"


def cand(label, amount, priority, line="", context=""):
    return LabelledCandidate(
        label=label, amount=Decimal(amount), line=line, priority=priority, context=context
    )


# A. Case-4-style: lower priority, weak winner -> ACCEPT override.
def test_case4_lower_priority_final_amount_is_accepted():
    context = ctx(
        winner=Decimal("3988.00"), winner_label="total amount", winner_priority=10,
        labelled=[
            cand("total amount", "3988.00", 10, line="Total Amount After Tax 3,988.00"),
            cand("total", "3988.40", 11, line="Total| 4.00 3,380.00 608.40 3,988.40"),
        ],
        candidates=[Decimal("3988.00"), Decimal("3988.40")],
    )
    raw = '{"decision":"override","amount":"3988.40","confidence":0.9,"evidence":"Total| ... 3,988.40"}'
    assert StubResolver(raw).resolve(context).amount == Decimal("3988.40")


# B. Case-8-style: lower priority, authoritative winner -> REJECT.
def test_case8_authoritative_winner_rejects_rollup():
    context = ctx(
        winner=Decimal("47925.00"), winner_label="invoice amount", winner_priority=0,
        labelled=[
            cand("invoice amount", "47925.00", 0, line="Invoice Amount: INR 47,925.00"),
            cand("total amount", "47924.68", 10, line="Total Amounts (INR) ... 47,924.68"),
        ],
        candidates=[Decimal("47925.00"), Decimal("47924.68")],
    )
    raw = '{"decision":"override","amount":"47924.68","confidence":1.0,"evidence":"Total Amounts 47,924.68"}'
    assert StubResolver(raw).resolve(context).amount is None


# C. Case-13-style: equal priority, tax context -> REJECT.
def test_case13_equal_priority_tax_context_rejected():
    context = ctx(
        winner=Decimal("1525.00"), winner_label="total", winner_priority=11,
        labelled=[
            cand("total", "1525.00", 11, line="TOTAL 15KG Rs. 1525.00", context="items | Discount"),
            cand(
                "total", "1500.00", 11,
                line="Total Rs. 1500.00 Rs. 37.50 Rs. 37.50 Rs. 75.00",
                context="CGST SGST | HSN Taxable Amount Total Tax Amount | Total Rs. 1500.00",
            ),
        ],
        candidates=[Decimal("1525.00"), Decimal("1500.00")],
    )
    raw = '{"decision":"override","amount":"1500.00","confidence":0.99,"evidence":"Total Rs. 1500.00"}'
    assert StubResolver(raw).resolve(context).amount is None


# D. Ungrounded amount -> REJECT.
def test_ungrounded_amount_rejected():
    context = ctx(
        winner=Decimal("1000.00"), winner_label="total amount", winner_priority=10,
        labelled=[cand("total", "900.00", 11)], candidates=[Decimal("1000.00"), Decimal("900.00")],
    )
    raw = '{"decision":"override","amount":"55555.00","confidence":0.99,"evidence":"x"}'
    assert StubResolver(raw).resolve(context).amount is None


# E. Low confidence -> REJECT.
def test_low_confidence_rejected():
    context = ctx(
        winner=Decimal("1000.00"), winner_label="total amount", winner_priority=10,
        labelled=[cand("total", "900.00", 11)], candidates=[Decimal("1000.00"), Decimal("900.00")],
    )
    raw = '{"decision":"override","amount":"900.00","confidence":0.5,"evidence":"Total 900"}'
    assert StubResolver(raw).resolve(context).amount is None


# F. Missing evidence -> REJECT.
def test_missing_evidence_rejected():
    context = ctx(
        winner=Decimal("1000.00"), winner_label="total amount", winner_priority=10,
        labelled=[cand("total", "900.00", 11)], candidates=[Decimal("1000.00"), Decimal("900.00")],
    )
    raw = '{"decision":"override","amount":"900.00","confidence":0.9,"evidence":""}'
    assert StubResolver(raw).resolve(context).amount is None


# G. Resolver timeout/error -> keep deterministic.
def test_timeout_keeps_deterministic():
    context = ctx(winner=Decimal("1000.00"), winner_label="total amount", winner_priority=10)
    assert StubResolver(exc=httpx.TimeoutException("slow")).resolve(context).amount is None


# H. Confident deterministic result -> resolver never called.
def test_confident_bypasses_resolver():
    stub = StubResolver('{"decision":"override","amount":"9999","confidence":1.0,"evidence":"x"}')
    extractor = InvoiceAmountExtractor(FakeOcrClient(text="Grand Total Rs. 18,750.00"), AmountNormalizer())
    service = InvoiceVerificationService(extractor, AmountValidator(), resolver=stub)
    result = service.verify(b"%PDF", PDF, Decimal("18750.00"))
    assert result.matched is True
    assert stub.calls == 0


# I. Lower-priority candidate with insufficient evidence (low confidence) -> REJECT.
def test_lower_priority_insufficient_evidence_rejected():
    context = ctx(
        winner=Decimal("1000.00"), winner_label="total amount", winner_priority=10,
        labelled=[cand("total", "900.00", 11)], candidates=[Decimal("1000.00"), Decimal("900.00")],
    )
    raw = '{"decision":"override","amount":"900.00","confidence":0.4,"evidence":"Total 900"}'
    assert StubResolver(raw).resolve(context).amount is None


# J. Higher-authority valid override -> ACCEPT.
def test_higher_authority_override_accepted():
    context = ctx(
        winner=Decimal("1000.00"), winner_label="total", winner_priority=11,
        labelled=[
            cand("total", "1000.00", 11, line="Total 1000"),
            cand("invoice amount", "1200.00", 0, line="Invoice Amount 1200"),
        ],
        candidates=[Decimal("1000.00"), Decimal("1200.00")],
    )
    raw = '{"decision":"override","amount":"1200.00","confidence":0.9,"evidence":"Invoice Amount 1200"}'
    assert StubResolver(raw).resolve(context).amount == Decimal("1200.00")


# Equal-priority CLEAN override still allowed (tie), tax check not triggered.
def test_equal_priority_clean_override_allowed():
    context = ctx(
        winner=Decimal("1000.00"), winner_label="grand total", winner_priority=2,
        labelled=[
            cand("grand total", "1000.00", 2, line="Grand Total 1000"),
            cand("grand total", "1200.00", 2, line="Grand Total 1200"),
        ],
        candidates=[Decimal("1000.00"), Decimal("1200.00")],
    )
    raw = '{"decision":"override","amount":"1200.00","confidence":0.9,"evidence":"Grand Total 1200"}'
    assert StubResolver(raw).resolve(context).amount == Decimal("1200.00")
