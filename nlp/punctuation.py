"""Utilities for restoring punctuation in raw transcripts."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PunctuationResult:
    """Structured result returned by :class:`PunctuationService`."""

    original_text: str
    punctuated_text: str
    model_used: Optional[str] = None


class PunctuationService:
    """Restores punctuation using a pretrained model when available.

    The service lazily loads ``deepmultilingualpunctuation.PunctuationModel`` when the
    package is installed. If the dependency is missing the service falls back to a
    lightweight rule-based approach that at least guarantees sentence-final
    punctuation.
    """

    _DEFAULT_MODEL_NAME = "kredor/punctuate-all"

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or self._DEFAULT_MODEL_NAME
        self._model = None
        self._load_error: Optional[Exception] = None

    def _ensure_model(self) -> None:
        if self._model is not None or self._load_error is not None:
            return

        try:
            from deepmultilingualpunctuation import PunctuationModel  # type: ignore

            self._model = PunctuationModel(model=self._model_name)
        except Exception as exc:  # pragma: no cover - defensive branch
            self._load_error = exc
            logger.warning(
                "Falling back to rule-based punctuation because the pretrained model "
                "could not be loaded: %s",
                exc,
            )

    @staticmethod
    def _fallback_punctuate(text: str) -> str:
        cleaned = " ".join(text.split())
        if not cleaned:
            return text

        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        punctuated_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if sentence[-1] not in ".!?":
                sentence += "."
            sentence = sentence[0].upper() + sentence[1:]
            punctuated_sentences.append(sentence)
        return " ".join(punctuated_sentences)

    def punctuate(self, transcript: str) -> PunctuationResult:
        """Return a :class:`PunctuationResult` with restored punctuation."""

        self._ensure_model()

        if self._model is not None:
            punctuated = self._model.restore_punctuation(transcript).strip()
            model_used = getattr(self._model, "model", self._model_name)
        else:
            punctuated = self._fallback_punctuate(transcript)
            model_used = None

        return PunctuationResult(
            original_text=transcript,
            punctuated_text=punctuated,
            model_used=model_used,
        )


__all__ = ["PunctuationService", "PunctuationResult"]
