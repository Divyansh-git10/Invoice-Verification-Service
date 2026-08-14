from typing import Optional

from app.extractors.ocr_client import OcrClient
from app.models.resolved_amount import ResolvedAmount
from app.resolvers.amount_resolver import AmountResolver, ResolverContext


class FakeOcrClient(OcrClient):
    """Test double for the OCR seam.

    Returns canned text (or raises a canned error) so the extractor's
    parsing/identification logic can be tested without Tesseract.
    """

    def __init__(self, text: str = "", raises: Exception | None = None):
        self._text = text
        self._raises = raises
        self.calls: list[tuple[int, str]] = []

    def extract_text(self, file_bytes: bytes, mime_type: str) -> str:
        self.calls.append((len(file_bytes), mime_type))
        if self._raises is not None:
            raise self._raises
        return self._text


class FakeAmountResolver(AmountResolver):
    """Test double for the AmountResolver seam. Returns a canned ResolvedAmount
    (or raises a canned error) and records calls, so the service's fallback
    orchestration can be tested without any LLM/network."""

    def __init__(
        self,
        result: Optional[ResolvedAmount] = None,
        raises: Optional[Exception] = None,
    ):
        self._result = result
        self._raises = raises
        self.calls: list[ResolverContext] = []

    def resolve(self, context: ResolverContext) -> ResolvedAmount:
        self.calls.append(context)
        if self._raises is not None:
            raise self._raises
        if self._result is not None:
            return self._result
        return ResolvedAmount(amount=None, confidence=0.0, evidence=None)
