"""Real-LLM evaluation over the 13 real invoices (Component 8).

Runs ONLY when GROQ_API_KEY is set. For each invoice it shows the deterministic
outcome, whether the LLM was escalated, the (grounded) LLM result, and the
final amount vs expected. Makes at most one LLM call per ambiguous/missing
invoice; confident invoices bypass the LLM entirely. Not collected by pytest.

Run:  python tests/real_invoices/eval_llm.py
"""
import mimetypes
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.extractors.invoice_amount_extractor import build_default_extractor
from app.resolvers.amount_resolver import ResolverContext
from app.resolvers.llm_amount_resolver import LlmAmountResolver

BASE = Path(__file__).resolve().parent

EXPECTED = [
    ("1.jpg", "105200.00"), ("2.jpg", "27904.40"), ("3.jpg", "4490.00"),
    ("4.jpg", "3988.40"), ("5.jpg", "30180.00"), ("6.jpg", "143370.00"),
    ("7.jpg", "968.00"), ("8.jpg", "47925.00"), ("9.jpg", "38026.00"),
    ("10.jpg", None), ("11.jpg", "4490.00"), ("12.jpg", "68230.50"),
    ("13.jpg", "1525.00"),
]


def _dec(v):
    if v is None:
        return None
    try:
        return Decimal(str(v))
    except InvalidOperation:
        return None


def _matches(expected, actual) -> bool:
    if expected is None:
        return actual is None
    return actual is not None and _dec(expected) == actual


def main() -> int:
    if not settings.GROQ_API_KEY:
        print("GROQ_API_KEY not set - skipping real-LLM evaluation. "
              "Set the key and re-run to evaluate.")
        return 0

    extractor = build_default_extractor()
    resolver = LlmAmountResolver(api_key=settings.GROQ_API_KEY)

    hdr = (f"{'inv':<7}{'det_amount':>13}{'det_conf':>9}{'llm':>5}"
           f"{'llm_amount':>13}{'llm_conf':>9}{'final':>13}{'expected':>13}  res")
    print(hdr)
    print("-" * len(hdr))

    det_pass = final_pass = recovered = unresolved = rejected = escalated = 0
    latencies = []

    for name, expected in EXPECTED:
        path = BASE / name
        mime, _ = mimetypes.guess_type(path.name)
        outcome = extractor.resolve(path.read_bytes(), mime)

        det_amount = outcome.amount
        det_ok = _matches(expected, det_amount)
        det_pass += det_ok

        llm_called = False
        llm_amount = None
        llm_conf = None

        if outcome.confident:
            final = det_amount
        else:
            llm_called = True
            escalated += 1
            context = ResolverContext(
                ocr_text=outcome.ocr_text,
                candidates=outcome.candidates,
                deterministic_winner=outcome.amount,
                winner_label=outcome.winner_label,
                winner_priority=outcome.winner_priority,
                labelled_candidates=outcome.labelled_candidates,
                amount_in_words=outcome.amount_in_words,
            )
            t0 = time.perf_counter()
            resolved = resolver.resolve(context)
            latencies.append(time.perf_counter() - t0)
            llm_amount = resolved.amount
            llm_conf = resolved.confidence
            if llm_amount is None:
                rejected += 1
                final = det_amount
            else:
                final = llm_amount

        final_ok = _matches(expected, final)
        final_pass += final_ok
        if final_ok and not det_ok:
            recovered += 1
        if not final_ok:
            unresolved += 1

        def s(x):
            return "None" if x is None else str(x)

        print(f"{name:<7}{s(det_amount):>13}{str(outcome.confident):>9}"
              f"{('yes' if llm_called else 'no'):>5}{s(llm_amount):>13}"
              f"{s(llm_conf):>9}{s(final):>13}{s(expected):>13}  "
              f"{'PASS' if final_ok else 'FAIL'}")

    print("-" * len(hdr))
    print(f"deterministic PASS: {det_pass}/13 | with-LLM PASS: {final_pass}/13")
    print(f"escalated to LLM: {escalated} | recovered: {recovered} | "
          f"still-unresolved: {unresolved} | LLM null/rejected: {rejected}")
    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"LLM latency per call: avg={avg:.2f}s min={min(latencies):.2f}s "
              f"max={max(latencies):.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
