"""Candidate-selection reframe: the LLM SELECTS one grounded candidate (or null)
and the deterministic gates decide accept/reject. The winner stays the safe
fallback; every existing gate is unchanged. No real LLM calls (provider stubbed).

Focused coverage for the six reframe cases:
  1. selected winner            -> keep (gate 5)
  2. selected_amount null       -> keep (null selection)
  3. valid better candidate     -> existing gates decide (accept)
  4. 8.jpg authority rejection  -> keep (authority veto)
  5. 13.jpg tax-context reject  -> keep (tax-context veto)
  6. ungrounded selected_amount -> keep (grounding gate)

Priority reference (lower = more authoritative):
  invoice amount=0, invoice total=1, grand total=2, ..., total amount=10, total=11
"""
from decimal import Decimal

from tests.test_llm_amount_resolver import StubResolver, ctx, lc


# 1. The LLM selects the deterministic winner itself -> gate 5 keeps it.
def test_selected_winner_is_kept():
    context = ctx(
        winner=Decimal("1000.00"), winner_label="invoice amount", winner_priority=0,
        labelled=[lc("invoice amount", "1000.00", 0)],
        candidates=[Decimal("1000.00")],
    )
    raw = '{"selected_amount":"1000.00","confidence":0.99,"evidence":"Invoice Amount 1000"}'
    assert StubResolver(raw).resolve(context).amount is None


# 2. A null selection keeps the deterministic winner (replaces the old keep decision).
def test_null_selection_is_kept():
    context = ctx(winner=Decimal("1000.00"), winner_label="invoice amount", winner_priority=0)
    raw = '{"selected_amount":null,"confidence":0.0,"evidence":null}'
    assert StubResolver(raw).resolve(context).amount is None


# 3. A grounded, better-supported candidate is selected -> existing gates accept it.
def test_valid_better_candidate_passes_gates():
    context = ctx(
        winner=Decimal("3988.00"), winner_label="total amount", winner_priority=10,
        labelled=[lc("total amount", "3988.00", 10), lc("total", "3988.40", 11)],
        candidates=[Decimal("3988.00"), Decimal("3988.40")],
    )
    # 4.jpg-style: winner label is weak (not authoritative) -> override allowed.
    raw = '{"selected_amount":"3988.40","confidence":0.9,"evidence":"Total ... 3,988.40"}'
    assert StubResolver(raw).resolve(context).amount == Decimal("3988.40")


# 4. 8.jpg: a lower-priority roll-up selected against an authoritative winner ->
#    the authority veto (gate 6) still rejects it.
def test_8jpg_authority_veto_still_rejects():
    context = ctx(
        winner=Decimal("47925.00"), winner_label="invoice amount", winner_priority=0,
        labelled=[lc("invoice amount", "47925.00", 0), lc("total amount", "47924.68", 10)],
        candidates=[Decimal("47925.00"), Decimal("47924.68")],
    )
    raw = '{"selected_amount":"47924.68","confidence":1.0,"evidence":"Total Amounts 47,924.68"}'
    assert StubResolver(raw).resolve(context).amount is None


# 5. 13.jpg: an equal-priority tax-section total selected -> the tax-context veto
#    (gate 6) still rejects it.
def test_13jpg_tax_context_veto_still_rejects():
    context = ctx(
        winner=Decimal("1525.00"), winner_label="total", winner_priority=11,
        labelled=[
            lc("total", "1525.00", 11, line="TOTAL 15KG Rs. 1525.00"),
            lc("total", "1500.00", 11,
               line="Total Tax Amount CGST SGST Rs. 1500.00 Rs. 37.50 Rs. 75.00"),
        ],
        candidates=[Decimal("1525.00"), Decimal("1500.00")],
    )
    raw = '{"selected_amount":"1500.00","confidence":0.99,"evidence":"Total Rs. 1500.00"}'
    assert StubResolver(raw).resolve(context).amount is None


# 6. An amount that is not among the grounded candidates -> the grounding gate
#    (gate 4) rejects it; the winner is kept.
def test_ungrounded_selection_is_kept():
    context = ctx(
        winner=Decimal("20000"), winner_label="total", winner_priority=11,
        labelled=[lc("total", "20000", 11)],
        candidates=[Decimal("20000")],
    )
    raw = '{"selected_amount":"19999.99","confidence":0.99,"evidence":"fabricated"}'
    assert StubResolver(raw).resolve(context).amount is None
