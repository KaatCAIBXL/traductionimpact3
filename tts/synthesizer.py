"""Sentence level text-to-speech synthesis with queue based streaming output."""

from __future__ import annotations

import asyncio
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Callable, Dict, List, Optional, Sequence


try:  # pragma: no cover - optional dependency
    import edge_tts  # type: ignore
except Exception:  # pragma: no cover - defensive fallback
    edge_tts = None  # type: ignore


__all__ = [
    "SentenceMarker",
    "SynthesisResult",
    "SynthesizedSentence",
    "SentenceSynthesizer",
]


SentenceGenerator = Callable[[str, str], "SynthesisResult"]
VoiceSelector = Callable[[str], str]


@dataclass
class SentenceMarker:
    """Lightweight marker describing sentence level boundaries."""

    kind: str
    offset_ms: Optional[float] = None
    payload: Dict[str, object] = field(default_factory=dict)


@dataclass
class SynthesisResult:
    """Result returned by the low level speech synthesis engine."""

    audio_path: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class _SentenceTask:
    """Internal queue payload describing work that needs to be synthesised."""

    block_id: str
    sentence_index: int
    sentence: str
    lang: str
    voice: str
    created_at: float


@dataclass
class SynthesizedSentence:
    """High level representation for consumers (UI, playback, analytics)."""

    block_id: str
    sentence_index: int
    sentence: str
    lang: str
    voice: str
    audio_path: Optional[str]
    markers: List[SentenceMarker] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    started_at: float = 0.0
    completed_at: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        """Serialize into a structure that can be JSON encoded."""

        return {
            "block_id": self.block_id,
            "sentence_index": self.sentence_index,
            "sentence": self.sentence,
            "lang": self.lang,
            "voice": self.voice,
            "audio_path": self.audio_path,
            "markers": [
                {
                    "kind": marker.kind,
                    "offset_ms": marker.offset_ms,
                    "payload": marker.payload,
                }
                for marker in self.markers
            ],
            "metadata": self.metadata,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


def _default_voice_selector(lang: str) -> str:
    return lang


def _default_sentence_generator(text: str, voice: str) -> "SynthesisResult":
    """Fallback synthesis implementation using edge-tts when available."""

    if edge_tts is None:
        # We return a stub result so the pipeline keeps working during tests.
        return SynthesisResult(audio_path=None, duration_ms=None, metadata={"engine": "stub"})

    communicator = edge_tts.Communicate(text, voice)

    async def _save() -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            await communicator.save(tmp.name)
            return tmp.name

    try:
        temp_file = asyncio.run(_save())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            temp_file = loop.run_until_complete(_save())
        finally:
            loop.close()
    # The library does not expose duration directly; we approximate using
    # the length of the text as a lightweight heuristic.
    estimated_duration = max(len(text.split()) * 320.0, 1000.0)
    return SynthesisResult(
        audio_path=temp_file,
        duration_ms=estimated_duration,
        metadata={"engine": "edge-tts"},
    )


class SentenceSynthesizer:
    """Queue based sentence level synthesiser with realtime results."""

    def __init__(
        self,
        *,
        voice_selector: Optional[VoiceSelector] = None,
        sentence_generator: Optional[SentenceGenerator] = None,
        result_queue_size: int = 64,
    ) -> None:
        self._voice_selector = voice_selector or _default_voice_selector
        self._sentence_generator = sentence_generator or _default_sentence_generator
        self._pending: "Queue[_SentenceTask]" = Queue()
        self._results: "Queue[SynthesizedSentence]" = Queue(maxsize=result_queue_size)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run, name="tts-worker", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._worker.join(timeout=1.0)

    def enqueue_block(
        self,
        sentences: Sequence[str],
        *,
        lang: str,
        block_id: Optional[str] = None,
    ) -> str:
        block_identifier = block_id or str(uuid.uuid4())
        voice = self._voice_selector(lang)
        created_at = time.time()
        for index, sentence in enumerate(sentence.strip() for sentence in sentences):
            if not sentence:
                continue
            task = _SentenceTask(
                block_id=block_identifier,
                sentence_index=index,
                sentence=sentence,
                lang=lang,
                voice=voice,
                created_at=created_at,
            )
            self._pending.put(task)
        return block_identifier

    def get_result(self, timeout: Optional[float] = None) -> Optional[SynthesizedSentence]:
        try:
            return self._results.get(timeout=timeout)
        except Empty:
            return None

    def drain_ready(self) -> List[SynthesizedSentence]:
        results: List[SynthesizedSentence] = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except Empty:
                break
        return results

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                task = self._pending.get(timeout=0.1)
            except Empty:
                continue

            started_at = time.time()
            synthesis = self._sentence_generator(task.sentence, task.voice)
            completed_at = time.time()
            duration_ms = synthesis.duration_ms
            if duration_ms is None:
                duration_ms = max((completed_at - started_at) * 1000.0, 0.0)

            marker = SentenceMarker(
                kind="sentence_end",
                offset_ms=duration_ms,
                payload={"sentence_index": task.sentence_index},
            )
            synthesized = SynthesizedSentence(
                block_id=task.block_id,
                sentence_index=task.sentence_index,
                sentence=task.sentence,
                lang=task.lang,
                voice=task.voice,
                audio_path=synthesis.audio_path,
                markers=[marker],
                metadata={**synthesis.metadata, "created_at": task.created_at},
                started_at=started_at,
                completed_at=completed_at,
            )

            try:
                self._results.put_nowait(synthesized)
            except Exception:
                # Queue full; drop the oldest result to ensure progress.
                try:
                    self._results.get_nowait()
                    self._results.put_nowait(synthesized)
                except Empty:
                    pass
            finally:
                self._pending.task_done()

        # Flush remaining pending tasks so the worker thread can exit cleanly.
        while True:
            try:
                self._pending.get_nowait()
                self._pending.task_done()
            except Empty:
                break
