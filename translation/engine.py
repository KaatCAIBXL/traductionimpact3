"""Translation utilities focused on block-based context aware translation."""

from __future__ import annotations

import os
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Optional, Sequence

import logging

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency guard
    import deepl
except Exception:  # pragma: no cover - optional dependency guard
    deepl = None  # type: ignore


def _build_default_translator() -> Optional["deepl.Translator"]:
    if deepl is None:
        return None
    api_key = os.getenv("DEEPL_API_KEY")
    if not api_key:
        logger.warning("DEEPL_API_KEY ontbreekt; er wordt geen automatische vertaling uitgevoerd.")
        return None
    try:
        return deepl.Translator(api_key)
    except Exception as exc:  # pragma: no cover - defensive branch
        logger.error("Kon DeepL vertaler niet initialiseren: %s", exc)
        return None


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    if not text.strip():
        return []
    parts = _SENTENCE_END_RE.split(text.strip())
    return [part.strip() for part in parts if part.strip()]


@dataclass
class TranslationContext:
    source_history: Deque[str] = field(default_factory=lambda: deque(maxlen=4))
    target_history: Deque[str] = field(default_factory=lambda: deque(maxlen=4))


class TranslationEngine:
    """Translate sentence blocks (1-4 sentences) with minimal context tracking."""

    def __init__(
        self,
        translator: Optional[object] = None,
        *,
        history_size: int = 4,
    ) -> None:
        self._translator = translator or _build_default_translator()
        self._context = TranslationContext(
            source_history=deque(maxlen=history_size),
            target_history=deque(maxlen=history_size),
        )

    @property
    def context(self) -> TranslationContext:
        return self._context

    def reset_context(self) -> None:
        self._context.source_history.clear()
        self._context.target_history.clear()

    def _translate_with_deepl(
        self,
        text: str,
        source_lang: Optional[str],
        target_lang: str,
    ) -> str:
        if self._translator is None:
            logger.debug("Geen vertaler beschikbaar; retourneer brontekst.")
            return text

        translate_func = getattr(self._translator, "translate_text", None)
        if translate_func is None:  # pragma: no cover - defensive branch
            logger.warning("Vertaler heeft geen translate_text methode; retourneer brontekst.")
            return text

        result = translate_func(text, source_lang=source_lang, target_lang=target_lang)
        translated_text = getattr(result, "text", result)
        return str(translated_text).strip()

    def translate_block(
        self,
        sentences: Sequence[str],
        *,
        source_lang: Optional[str],
        target_lang: str,
        extra_context: Optional[Iterable[str]] = None,
    ) -> str:
        """Translate a block (1-4 sentences) while keeping contextual history."""

        if not sentences:
            return ""
        if len(sentences) > 4:
            raise ValueError("Een vertaalblok mag maximaal 4 zinnen bevatten.")

        block_text = " ".join(sentence.strip() for sentence in sentences if sentence.strip())
        if not block_text:
            return ""

        context_segments: List[str] = list(self._context.source_history)
        if extra_context:
            context_segments.extend(extra_context)
        context_text = " ".join(context_segments)

        payload = f"{context_text}\n\n{block_text}" if context_text else block_text
        translated_text = self._translate_with_deepl(payload, source_lang, target_lang)

        if context_text and translated_text:
            # Probeer enkel het blok uit de vertaling te halen door de laatste alineas te nemen.
            paragraphs = [part.strip() for part in translated_text.splitlines() if part.strip()]
            if paragraphs:
                translated_text = paragraphs[-1]

        self._context.source_history.append(block_text)
        self._context.target_history.append(translated_text)

        return translated_text

    def translate_block_from_text(
        self,
        block_text: str,
        *,
        source_lang: Optional[str],
        target_lang: str,
    ) -> str:
        sentences = _split_sentences(block_text)
        return self.translate_block(sentences, source_lang=source_lang, target_lang=target_lang)


__all__ = ["TranslationEngine", "TranslationContext", "_split_sentences"]
