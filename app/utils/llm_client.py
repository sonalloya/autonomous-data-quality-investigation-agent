"""
app/utils/llm_client.py
-----------------------
Phase 3 — Lightweight LLM Client for Google Gemini (google-genai SDK).

Wraps google-genai with graceful fallback when:
  - The SDK is not installed
  - LLM_API_KEY is not set
  - The API call fails

All callers should check `is_available()` before calling `generate()`.
Never log or expose the API key.

Usage:
    from app.utils.llm_client import llm_client

    if llm_client.is_available():
        reply = llm_client.generate(system_prompt, user_message)
    else:
        reply = None   # fall back to deterministic output
"""

from __future__ import annotations

import logging
from typing import Optional

from app.utils.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing the Gemini SDK; set a flag if unavailable
# ---------------------------------------------------------------------------
try:
    from google import genai as _genai          # type: ignore
    from google.genai import types as _gtypes   # type: ignore
    _GEMINI_AVAILABLE = True
except ImportError:
    _genai   = None   # type: ignore
    _gtypes  = None   # type: ignore
    _GEMINI_AVAILABLE = False


class LLMClient:
    """
    Thin wrapper around the Google Gemini API (google-genai SDK v2+).

    Instantiated once as a module-level singleton (``llm_client``).
    Callers must check ``is_available()`` before calling ``generate()``.
    """

    _DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(self) -> None:
        self._configured = False
        self._model_name = ""
        self._client     = None   # google.genai.Client

        if not _GEMINI_AVAILABLE:
            logger.info(
                "[LLMClient] google-genai SDK not installed. "
                "LLM interpretation will be skipped."
            )
            return

        api_key = settings.LLM_API_KEY
        if not api_key:
            logger.info(
                "[LLMClient] LLM_API_KEY not set. "
                "LLM interpretation will be skipped."
            )
            return

        try:
            self._client     = _genai.Client(api_key=api_key)   # type: ignore[union-attr]
            self._model_name = settings.LLM_MODEL or self._DEFAULT_MODEL
            self._configured = True
            logger.info(
                "[LLMClient] Gemini client configured with model '%s'.",
                self._model_name,
            )
        except Exception as exc:
            logger.warning("[LLMClient] Gemini configuration failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the LLM can be called."""
        return self._configured and self._client is not None

    def model_name(self) -> str:
        """Return the model identifier currently in use."""
        return self._model_name or "not-configured"

    def generate(
        self,
        system_prompt:     str,
        user_message:      str,
        max_output_tokens: int = 1024,
    ) -> Optional[str]:
        """
        Call the Gemini API and return the text response.

        Returns ``None`` on any failure so callers can gracefully fall back.

        Parameters
        ----------
        system_prompt : str
            Instructions for the model (role/behaviour).
        user_message : str
            The actual content to process.
        max_output_tokens : int
            Upper bound on response length.
        """
        if not self.is_available():
            return None

        # Combine system and user into a single content string (flash/pro approach)
        combined = f"{system_prompt}\n\n---\n\n{user_message}"

        try:
            response = self._client.models.generate_content(   # type: ignore[union-attr]
                model=self._model_name,
                contents=combined,
                config=_gtypes.GenerateContentConfig(          # type: ignore[union-attr]
                    max_output_tokens=max_output_tokens,
                    temperature=0.2,
                ),
            )
            text = response.text.strip() if response.text else ""
            return text if text else None
        except Exception as exc:
            logger.warning("[LLMClient] generate() failed: %s", exc)
            return None


# Module-level singleton — import this wherever LLM access is needed
llm_client = LLMClient()
