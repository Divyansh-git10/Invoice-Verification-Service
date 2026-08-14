import json
from decimal import Decimal
from typing import Optional

import httpx
from pydantic import ValidationError

from app.core.config import settings
from app.core.logger import get_logger
from app.models.resolved_amount import ResolvedAmount
from app.resolvers.amount_resolver import AmountResolver, ResolverContext
from app.utils.amount_normalizer import AmountNormalizer

logger = get_logger(__name__)

_KEEP = ResolvedAmount(amount=None, confidence=0.0, evidence=None)

# Explicit invoice-level final-total labels. When the deterministic winner
# already carries one of these, a strictly lower-priority override is refused.
_AUTHORITATIVE_WINNER_LABELS = frozenset(
    {
        "invoice amount",
        "invoice total",
        "grand total",
        "total amount payable",
        "amount payable",
        "net payable",
    }
)

# Markers that identify a tax / breakdown section. Used only to refuse an
# equal-priority override whose local context is clearly a tax summary.
_TAX_CONTEXT_MARKERS = (
    "cgst",
    "sgst",
    "igst",
    "taxable",
    "total tax",
    "tax amount",
    "hsn",
    "sac",
)

_SYSTEM_PROMPT = (
    "You audit a deterministic extractor's chosen invoice total. The "
    "deterministic winner is the DEFAULT and is usually correct. Return "
    'decision "keep" unless the OCR clearly shows that a DIFFERENT amount is '
    "the FINAL PAYABLE amount - the amount the customer must ultimately pay.\n"
    "When identifying the final payable amount, prefer in order:\n"
    "1. Grand Total / Final Total / Amount Payable\n"
    "2. Invoice Amount / Invoice Total\n"
    "3. Total Amount After Tax\n"
    "4. An explicit invoice-level Total\n"
    "5. Amount in words referring to the invoice total\n"
    "Never select: subtotal, taxable amount, total tax, CGST/SGST/IGST, "
    "tax-section totals, item totals, discount, advance, previous balance, "
    "payment/received amounts, or intermediate calculated totals.\n"
    "When several \"Total\" values exist, choose the one in the invoice-level "
    "final-payable section, not a tax or item breakdown. Do not just pick the "
    "largest amount or any amount labelled \"Total\".\n"
    "The amount being selected must represent the invoice-level FINAL PAYABLE "
    "amount. A candidate being labelled 'Total' is not sufficient by itself. "
    "Each candidate includes its surrounding OCR context; use that context and "
    "section/location clues to distinguish the invoice-level total from tax "
    "summaries, subtotals, item totals, and intermediate totals.\n"
    "Rules:\n"
    "- Use only amounts that literally appear in the OCR text. Never invent or "
    "correct digits. Do not do arithmetic. Do not fuzzy-match.\n"
    "- A bare monetary number is NOT sufficient to override. An override must "
    "be supported by an explicit labelled final-total candidate.\n"
    "- If uncertain, or if the winner is already the final payable total, "
    'return "keep".\n'
    'Respond ONLY as JSON: {"decision": "keep" | "override", '
    '"amount": "<decimal string>" | null, "confidence": <0..1>, '
    '"evidence": "<exact OCR quote>" | null}. '
    'For "keep": amount=null and evidence=null. For "override": amount and '
    "evidence are required."
)


class LlmAmountResolver(AmountResolver):
    """Groq-backed arbitrator (default llama-3.1-8b-instant) behind the
    AmountResolver seam. One JSON-mode call, short timeout, no retries. The
    deterministic winner is kept unless every override gate passes. Any failure
    degrades to keep (amount=None). Provider specifics stay isolated here."""

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        base_url: Optional[str] = None,
        normalizer: Optional[AmountNormalizer] = None,
        min_confidence: Optional[float] = None,
    ):
        self._api_key = api_key
        self._model = model or settings.LLM_MODEL
        self._timeout = timeout if timeout is not None else settings.LLM_TIMEOUT_SECONDS
        self._base_url = (base_url or settings.GROQ_BASE_URL).rstrip("/")
        self._normalizer = normalizer or AmountNormalizer()
        self._min_confidence = (
            min_confidence
            if min_confidence is not None
            else settings.LLM_OVERRIDE_MIN_CONFIDENCE
        )

    def resolve(self, context: ResolverContext) -> ResolvedAmount:
        try:
            raw = self._request_completion(self._build_prompt(context))
        except Exception:  # noqa: BLE001 - network/timeout/HTTP errors -> keep
            logger.warning("LLM request failed; keeping deterministic result", exc_info=True)
            return _KEEP
        return self._arbitrate(raw, context)

    # -- provider call (isolated) --------------------------------------

    def _request_completion(self, user_prompt: str) -> str:
        payload = {
            "model": self._model,
            "temperature": 0,
            "max_tokens": 220,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        response = httpx.post(
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    # -- prompt --------------------------------------------------------

    def _build_prompt(self, context: ResolverContext) -> str:
        winner = (
            "none"
            if context.deterministic_winner is None
            else f"{context.deterministic_winner} (label: {context.winner_label})"
        )
        labelled = [
            {
                "label": c.label,
                "priority": c.priority,
                "amount": str(c.amount),
                "is_deterministic_winner": (
                    context.deterministic_winner is not None
                    and c.amount == context.deterministic_winner
                ),
                "context": c.context or c.line,
            }
            for c in context.labelled_candidates
        ]
        words = "none" if context.amount_in_words is None else str(context.amount_in_words)
        return (
            f"deterministic_winner: {winner}\n"
            f"labelled_total_candidates: {json.dumps(labelled)}\n"
            f"amount_in_words: {words}\n"
            f"OCR text:\n{context.ocr_text}\n"
            "Return JSON."
        )

    # -- deterministic arbitration gates -------------------------------

    def _arbitrate(self, raw: str, context: ResolverContext) -> ResolvedAmount:
        # Gate 1 - only an explicit override decision can replace the winner.
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.info("LLM returned malformed JSON; keeping deterministic result")
            return _KEEP
        if not isinstance(data, dict) or data.get("decision") != "override":
            return _KEEP

        amount_raw = data.get("amount")
        if amount_raw is None:
            return _KEEP

        # Gate 3 - normalize the proposed amount.
        try:
            amount = self._normalizer.normalize(str(amount_raw))
        except ValueError:
            return _KEEP

        # Gate 2 - structured validation (confidence range + evidence present).
        try:
            proposed = ResolvedAmount(
                amount=amount,
                confidence=float(data.get("confidence", 0.0)),
                evidence=data.get("evidence"),
            )
        except (ValidationError, ValueError, TypeError):
            return _KEEP

        # Gate 4 - grounding: the amount must be a monetary token in the OCR.
        if amount not in set(context.candidates):
            logger.info("Override %s not grounded in OCR candidates; keeping", amount)
            return _KEEP

        # Gate 7 - confidence threshold.
        if proposed.confidence < self._min_confidence:
            return _KEEP

        if context.deterministic_winner is not None:
            # Gate 5 - agreement is not an override.
            if amount == context.deterministic_winner:
                return _KEEP
            # Gate 6 - contextual override policy (replaces the absolute
            # label-priority veto).
            if not self._override_allowed(amount, context):
                logger.info("Override %s rejected by contextual policy; keeping", amount)
                return _KEEP
        else:
            # MISSING case: only a labelled candidate or the amount-in-words
            # value may resolve; a bare monetary token is not sufficient.
            if not self._supported_when_missing(amount, context):
                logger.info("No labelled/words support for %s with no winner; keeping", amount)
                return _KEEP

        return proposed

    def _override_allowed(self, amount: Decimal, context: ResolverContext) -> bool:
        """Contextual override policy. Label priority is preferred but is not an
        absolute veto: a lower-priority candidate may override when the winner
        is not itself an explicit invoice-level final label; an equal-priority
        candidate is refused only when its local context is clearly a tax
        breakdown. No fuzzy matching, no arithmetic, no positional heuristics."""
        cand = self._best_candidate_for(amount, context)
        if cand is None or context.winner_priority is None:
            return False

        if cand.priority < context.winner_priority:
            return True  # strictly higher authority than the winner

        if cand.priority == context.winner_priority:
            # Equal authority: refuse only a clearly tax/intermediate section.
            return not self._is_tax_context(cand)

        # Lower authority: allowed only when the deterministic winner is not
        # itself an explicit authoritative invoice-level final label.
        return (context.winner_label or "") not in _AUTHORITATIVE_WINNER_LABELS

    @staticmethod
    def _best_candidate_for(amount: Decimal, context: ResolverContext):
        matches = [c for c in context.labelled_candidates if c.amount == amount]
        return min(matches, key=lambda c: c.priority) if matches else None

    @staticmethod
    def _is_tax_context(candidate) -> bool:
        blob = f"{candidate.line} {candidate.context}".lower()
        return any(marker in blob for marker in _TAX_CONTEXT_MARKERS)

    def _supported_when_missing(self, amount: Decimal, context: ResolverContext) -> bool:
        if context.amount_in_words is not None and amount == context.amount_in_words:
            return True
        return any(c.amount == amount for c in context.labelled_candidates)


def build_default_resolver() -> Optional[AmountResolver]:
    """Return a Groq resolver when GROQ_API_KEY is configured, else None so the
    pipeline stays deterministic-only. No network call at construction time."""
    if not settings.GROQ_API_KEY:
        return None
    return LlmAmountResolver(api_key=settings.GROQ_API_KEY)
