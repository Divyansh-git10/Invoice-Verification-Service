from app.api.response import VerificationResponse
from app.models.validation_result import ValidationResult


class ResponseMapper:

    @staticmethod
    def to_response(
        result: ValidationResult,
    ) -> VerificationResponse:
        return VerificationResponse(
            matched=result.matched,
            expected_amount=result.expected_amount,
            actual_amount=result.actual_amount,
        )
