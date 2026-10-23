"""Stable public errors for API responses and background job records."""
from __future__ import annotations

from fastapi import HTTPException


def public_error(exc: Exception) -> dict[str, str]:
    status = getattr(exc, "status_code", getattr(exc, "code", None))
    detail = str(exc).lower()
    if isinstance(exc, HTTPException):
        if status == 404:
            return {"code": "NOT_FOUND", "message": str(exc.detail),
                    "suggested_action": "Check the document or job ID."}
        if status == 409:
            return {"code": "NOT_READY", "message": str(exc.detail),
                    "suggested_action": "Wait for indexing to complete and retry."}
        return {"code": "INVALID_REQUEST", "message": str(exc.detail),
                "suggested_action": "Check the request and try again."}
    if status == 429:
        if any(term in detail for term in (
                "insufficient_quota", "credit_balance_exhausted",
                "no credits remaining")):
            return {"code": "BILLING_REQUIRED",
                    "message": "The AI provider has no API credits remaining.",
                    "suggested_action": "Add API credits or select a local provider."}
        return {"code": "RATE_LIMITED", "message": "AI provider is limiting requests.",
                "suggested_action": "Retry later or select a local model."}
    if isinstance(status, int) and status >= 500:
        return {"code": "UPSTREAM_UNAVAILABLE",
                "message": "The AI provider is temporarily unavailable.",
                "suggested_action": "Retry later or configure a fallback provider."}
    if isinstance(exc, (LookupError, FileNotFoundError)):
        return {"code": "NOT_FOUND", "message": "The requested document was not found.",
                "suggested_action": "Check the document ID and upload status."}
    if isinstance(exc, ValueError):
        return {"code": "INVALID_DOCUMENT", "message": str(exc)[:300],
                "suggested_action": "Check the document format and content."}
    if isinstance(exc, RuntimeError) and "embedding" in str(exc).lower():
        return {"code": "MODEL_CONFIGURATION", "message": str(exc)[:300],
                "suggested_action": "Check the embedding model and Qdrant collection."}
    return {"code": "PROCESSING_ERROR",
            "message": "The request could not be completed.",
            "suggested_action": "Check server logs and retry."}
