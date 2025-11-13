"""Transcription manager that keeps a rolling transcript of processed audio segments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

from audio_processing import AudioSegment


@dataclass
class TranscriptSegment:
    """Represents a single transcription segment with timing information."""

    raw_text: str
    start_time: float
    end_time: float
    raw_audio: Optional[AudioSegment] = None

    def as_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class TranscriptionManager:
    """Store and maintain a rolling transcript of speech segments."""

    def __init__(self, *, max_segments: Optional[int] = 500) -> None:
        self.max_segments = max_segments
        self._segments: List[TranscriptSegment] = []
        self._listeners: List[Callable[[TranscriptSegment], None]] = []

    # ------------------------------------------------------------------
    # Listener registration
    def add_listener(self, listener: Callable[[TranscriptSegment], None]) -> None:
        """Register a callback that receives every new transcript segment."""

        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[TranscriptSegment], None]) -> None:
        """Remove a previously registered listener."""

        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Segment management
    def add_segment(self, raw_text: str, *, start_time: float, end_time: float, raw_audio: Optional[AudioSegment] = None) -> TranscriptSegment:
        """Append a new segment to the rolling transcript."""

        segment = TranscriptSegment(raw_text=raw_text, start_time=start_time, end_time=end_time, raw_audio=raw_audio)
        self._segments.append(segment)
        self._prune_if_needed()
        for listener in list(self._listeners):
            listener(segment)
        return segment

    def consume_audio_segment(self, segment: AudioSegment, *, transcribe: Optional[Callable[[AudioSegment], str]] = None) -> TranscriptSegment:
        """Transcribe an :class:`~audio_processing.AudioSegment` and store the result.

        Parameters
        ----------
        segment:
            Audio segment produced by :class:`audio_processing.MicrophoneBuffer`.
        transcribe:
            Optional callable that turns ``segment`` into text.  When omitted an empty
            string is stored for ``text`` so downstream consumers can fill it later.
        """

        raw_text = transcribe(segment) if transcribe else ""
        return self.add_segment(raw_text, start_time=segment.start_time, end_time=segment.end_time, raw_audio=segment)

    def iter_segments(self) -> Iterable[TranscriptSegment]:
        """Return a snapshot iterator of the stored segments."""

        return list(self._segments)

    def get_full_transcript(self, *, separator: str = " ") -> str:
        """Join the stored text fragments into a single transcript string."""

        return separator.join(segment.raw_text for segment in self._segments if segment.raw_text)

    def clear(self) -> None:
        """Remove all stored segments."""

        self._segments.clear()

    def _prune_if_needed(self) -> None:
        if self.max_segments is None:
            return
        overflow = len(self._segments) - self.max_segments
        if overflow > 0:
            del self._segments[0:overflow]

    # ------------------------------------------------------------------
    # Serialization helpers
    def to_dict(self) -> List[dict]:
        return [segment.as_dict() for segment in self._segments]
