"""Corroborated amount-in-words vs. a competing item-table grand total.

When `amount_in_words` conflicts with a grounded candidate that is a
structurally-identified item-table Total-column value (`in_item_table_total`),
the corroborated block exposes `competes_with_table_total` / listing metadata so
the LLM compares both instead of treating the words value as decisive. This is
prompt/metadata only: no gate, deterministic winner, or arithmetic changes. No
real LLM calls (provider stubbed).
"""
import json
from decimal import Decimal

from app.extractors.invoice_amount_extractor import LabelledCandidate
from app.resolvers.llm_amount_resolver import LlmAmountResolver, _SYSTEM_PROMPT
from tests.test_llm_amount_resolver import StubResolver, ctx, lc

PDF = "application/pdf"


def _prompt(context) -> str:
    return LlmAmountResolver(api_key="x")._build_prompt(context)


def _corroborated_json(prompt: str) -> dict:
    for line in prompt.splitlines():
        if line.startswith("corroborated_amount_in_words:"):
            payload = line.split("corroborated_amount_in_words:", 1)[1].strip()
            return json.loads(payload)
    raise AssertionError("corroborated block not found")


def _tt(label, amount, priority, line="line"):
    return LabelledCandidate(
        label=label, amount=Decimal(amount), line=line, priority=priority,
        in_table_total=True,
    )


# --- (A) block metadata ----------------------------------------------------

def test_4jpg_style_words_with_competing_table_total_flags_true():
    # 3988.00 words (footer) competes with 3988.40 item-table Total column.
    context = ctx(
        winner=Decimal("3988.00"), winner_label="total amount", winner_priority=10,
        labelled=[
            lc("total amount", "3988.00", 10, line="Total Amount After Tax 3,988.00"),
            _tt("total", "3988.40", 11, line="Total 4.00 3,380.00 608.40 3,988.40"),
        ],
        candidates=[Decimal("3988.00"), Decimal("3988.40")],
        words=Decimal("3988.00"),
        ocr="Total 4.00 3,380.00 608.40 3,988.40\nTotal Amount After Tax 3,988.00\n",
    )
    block = _corroborated_json(_prompt(context))
    assert block["competes_with_table_total"] is True
    assert block["competing_table_total_candidates"] == ["3988.40"]
    # existing fields still present / unchanged
    assert block["value"] == "3988.00"
    assert block["cross_grounded"] is True


def test_7jpg_style_words_no_competing_table_total_flags_false():
    # Words value with no item-table total competing -> false / empty list.
    context = ctx(
        winner=Decimal("965.00"), winner_label="total", winner_priority=11,
        labelled=[lc("total", "965.00", 11, line="Total: 965.00")],
        candidates=[Decimal("965.00"), Decimal("968.00")],
        words=Decimal("968.00"),
        ocr="Total: 965.00\nAmount: 968.00\nNine Hundred Sixty-eight Only\n",
    )
    block = _corroborated_json(_prompt(context))
    assert block["competes_with_table_total"] is False
    assert block["competing_table_total_candidates"] == []


def test_table_total_equal_to_words_does_not_compete():
    # An in_table_total candidate whose amount EQUALS the words value is not a
    # competitor (exact inequality only).
    context = ctx(
        winner=Decimal("1000.00"), winner_label="total", winner_priority=11,
        labelled=[_tt("total", "1000.00", 11, line="Total ... 1000.00")],
        candidates=[Decimal("1000.00")],
        words=Decimal("1000.00"),
        ocr="Total ... 1000.00\nOne Thousand Only\n",
    )
    block = _corroborated_json(_prompt(context))
    assert block["competes_with_table_total"] is False
    assert block["competing_table_total_candidates"] == []


def test_absent_corroboration_still_renders_none():
    context = ctx(winner=Decimal("500.00"), winner_label="total", winner_priority=11, words=None)
    assert "corroborated_amount_in_words: none" in _prompt(context)


# --- (B) system prompt caveat ----------------------------------------------

def test_system_prompt_documents_competing_table_total_caveat():
    assert "competes_with_table_total" in _SYSTEM_PROMPT
    assert "competing_table_total_candidates" in _SYSTEM_PROMPT
    assert "do NOT treat the amount-in-words value as decisive" in _SYSTEM_PROMPT
    assert "A larger magnitude is not, by itself, a reason to prefer either" in _SYSTEM_PROMPT


# --- gates unchanged: 8.jpg authority veto, 13.jpg tax-context veto ---------

def test_8jpg_authority_veto_unchanged_even_with_competing_metadata():
    # 8.jpg: words == winner (47925.00); rollup 47924.68 is an in_table_total
    # candidate -> metadata may flag competition, but the authority veto still
    # rejects 47924.68.
    context = ctx(
        winner=Decimal("47925.00"), winner_label="invoice amount", winner_priority=0,
        labelled=[
            lc("invoice amount", "47925.00", 0, line="Invoice Amount: INR 47,925.00"),
            _tt("total amount", "47924.68", 10, line="Total Amounts (INR) ... 47,924.68"),
        ],
        candidates=[Decimal("47925.00"), Decimal("47924.68")],
        words=Decimal("47925.00"),
        ocr="Invoice Amount: INR 47,925.00\nTotal Amounts (INR) 38,991.00 8,933.68 47,924.68\n",
    )
    block = _corroborated_json(_prompt(context))
    assert block["competes_with_table_total"] is True          # metadata appears
    assert block["competing_table_total_candidates"] == ["47924.68"]
    # gate: LLM selecting the rollup is still rejected by the authority veto.
    raw = '{"selected_amount":"47924.68","confidence":1.0,"evidence":"Total Amounts 47,924.68"}'
    assert StubResolver(raw).resolve(context).amount is None


def test_13jpg_tax_context_veto_unchanged():
    # 13.jpg has no amount-in-words -> block renders none; tax-context veto still
    # rejects the equal-priority tax total 1500.00.
    context = ctx(
        winner=Decimal("1525.00"), winner_label="total", winner_priority=11,
        labelled=[
            lc("total", "1525.00", 11, line="TOTAL 15KG Rs. 1525.00"),
            _tt("total", "1500.00", 11,
                line="Total Tax Amount CGST SGST Rs. 1500.00 Rs. 37.50 Rs. 75.00"),
        ],
        candidates=[Decimal("1525.00"), Decimal("1500.00")],
        words=None,
    )
    assert "corroborated_amount_in_words: none" in _prompt(context)
    raw = '{"selected_amount":"1500.00","confidence":0.99,"evidence":"Total Rs. 1500.00"}'
    assert StubResolver(raw).resolve(context).amount is None


# --- 4.jpg: gate already allows the table total if the LLM selects it -------

def test_4jpg_table_total_override_still_allowed_by_existing_gate():
    context = ctx(
        winner=Decimal("3988.00"), winner_label="total amount", winner_priority=10,
        labelled=[
            lc("total amount", "3988.00", 10, line="Total Amount After Tax 3,988.00"),
            _tt("total", "3988.40", 11, line="Total 4.00 3,380.00 608.40 3,988.40"),
        ],
        candidates=[Decimal("3988.00"), Decimal("3988.40")],
        words=Decimal("3988.00"),
    )
    raw = '{"selected_amount":"3988.40","confidence":0.9,"evidence":"Total ... 3,988.40"}'
    assert StubResolver(raw).resolve(context).amount == Decimal("3988.40")
