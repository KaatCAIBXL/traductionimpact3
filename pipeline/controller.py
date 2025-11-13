"""High level orchestration for the speech-to-translation pipeline."""
The controller acts as the rendezvous point between transcription, text to
speech and the real-time UI. Each translated block receives a unique
identifier and timestamp, allowing downstream systems to synchronise audio
playback with text highlighting while sentences are streamed out of the TTS
queue.
"""

from __future__ import annotations
import time
import uuid

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

from nlp.punctuation import PunctuationResult, PunctuationService
from translation.engine import TranslationEngine, _split_sentences


TTSCallable = Callable[[str, Sequence[str], str, float], object]
UICallable = Callable[[Dict[str, object]], None]


@dataclass
class PipelineOutput:
    punctuated_text: str
    translated_text: str
    tts_payload: Optional[object]
    block_id: str
    block_started_at: float
    source_sentences: Sequence[str]
    translated_sentences: Sequence[str]


class PipelineController:
    """Runs the punctuator and translator before handing over to TTS/UI."""

     A fresh ``block_id`` and ``block_started_at`` timestamp are created for
        each call. These values travel to both the text-to-speech queue and the
        UI callback so that sentence level events can be synchronised: the TTS
        worker emits markers (``sentence_end``) that correspond with
        ``translated_sentences[index]`` while the display manager highlights the
        matching DOM node in real time.
        """

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

        block_id = str(uuid.uuid4())
        block_started_at = time.time()
        translated_sentences = _split_sentences(translated)


        tts_payload = None
        if self._tts_callback:
            tts_payload = self._tts_callback(
                block_id,
                translated_sentences or [translated],
                target_lang,
                block_started_at,
            )

        if self._ui_callback:
            self._ui_callback(
                {
                    "block_id": block_id,
                    "block_started_at": block_started_at,
                    "source": punctuated.punctuated_text,
                    "source_sentences": sentences,
                    "translation": translated,
                    "translated_sentences": translated_sentences,
                    "target_lang": target_lang,
                }
            )

        return PipelineOutput(
            punctuated_text=punctuated.punctuated_text,
            translated_text=translated,
            tts_payload=tts_payload,
            block_id=block_id,
            block_started_at=block_started_at,
            source_sentences=sentences,
            translated_sentences=translated_sentences,
        )


__all__ = ["PipelineController", "PipelineOutput"]
