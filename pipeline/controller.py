"""High level orchestration for the speech-to-translation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from nlp.punctuation import PunctuationResult, PunctuationService
from translation.engine import TranslationEngine, _split_sentences


TTSCallable = Callable[[str, str], object]
UICallable = Callable[[Dict[str, object]], None]


@dataclass
class PipelineOutput:
    punctuated_text: str
    translated_text: str
    tts_payload: Optional[object]


class PipelineController:
    """Runs the punctuator and translator before handing over to TTS/UI."""

    def __init__(
        self,
        *,
        punctuator: Optional[PunctuationService] = None,
        translator: Optional[TranslationEngine] = None,
        tts_callback: Optional[TTSCallable] = None,
        ui_callback: Optional[UICallable] = None,
    ) -> None:
        self._punctuator = punctuator or PunctuationService()
        self._translator = translator or TranslationEngine()
        self._tts_callback = tts_callback
        self._ui_callback = ui_callback

    def process_block(
        self,
        transcript_block: str,
        *,
        source_lang: Optional[str],
        target_lang: str,
    ) -> PipelineOutput:
        """Punctuate, translate and forward the result to TTS/UI."""

        punctuated: PunctuationResult = self._punctuator.punctuate(transcript_block)
        sentences = _split_sentences(punctuated.punctuated_text)
        translated = self._translator.translate_block(
            sentences,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        tts_payload = None
        if self._tts_callback:
            tts_payload = self._tts_callback(translated, target_lang)

        if self._ui_callback:
            self._ui_callback(
                {
                    "source": punctuated.punctuated_text,
                    "translation": translated,
                    "target_lang": target_lang,
                }
            )

        return PipelineOutput(
            punctuated_text=punctuated.punctuated_text,
            translated_text=translated,
            tts_payload=tts_payload,
        )


__all__ = ["PipelineController", "PipelineOutput"]
