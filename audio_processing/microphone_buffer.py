"""Microphone buffering and energy-based voice activity detection utilities.

This module exposes building blocks that make it simple to buffer microphone input and
emit audio segments once speech has ended.  The implementation is intentionally
framework-agnostic so it can be reused from CLI tools or async frameworks alike.

Example
-------

>>> from transcription.manager import TranscriptionManager
>>> from audio_processing import MicrophoneBuffer
>>>
>>> manager = TranscriptionManager()
>>> buffer = MicrophoneBuffer(on_segment_ready=manager.consume_audio_segment)
>>> buffer.add_audio_frame(raw_bytes)

When a segment finishes (detected via the energy-based VAD) the transcription manager
is notified.  The manager can then hand the segment to a speech-to-text backend and
store the resulting text together with the timecodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import time


@dataclass
class AudioSegment:
    """Container for a completed audio segment.

    Attributes
    ----------
    samples:
        A one-dimensional ``numpy`` array containing the raw PCM samples in ``int16``
        format.
    sample_rate:
        The sampling rate of ``samples``.
    start_time:
        Timestamp (in seconds) marking when the segment started.
    end_time:
        Timestamp (in seconds) marking when the segment ended.
    """

    samples: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float

    def as_bytes(self) -> bytes:
        """Return the segment as raw little-endian ``int16`` PCM bytes."""

        return self.samples.astype(np.int16).tobytes()

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""

        if self.samples.size == 0:
            return 0.0
        return self.samples.size / float(self.sample_rate)


class EnergyVADSegmenter:
    """Energy-based voice activity detector with hysteresis.

    Parameters
    ----------
    sample_rate:
        Sample rate of the incoming audio stream.
    frame_size:
        Size of frames (in samples) that ``process_frame`` expects.  The detector works
        with arbitrary frame sizes but performance is best when frames are short
        (~10-30ms of audio).
    energy_threshold:
        Frames whose RMS energy is below this threshold are treated as silence.  The
        value is normalized so ``1.0`` corresponds to the maximum possible energy for
        ``int16`` samples.
    silence_duration:
        Amount of time (seconds) that has to pass in silence before we commit the
        buffered audio as a completed segment.
    hangover_frames:
        Additional number of speech frames to keep after energy drops below the
        threshold.  This mirrors the "hangover" behaviour of VADs such as WebRTC.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        frame_size: int = 480,
        energy_threshold: float = 0.02,
        silence_duration: float = 0.5,
        hangover_frames: int = 3,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.hangover_frames = hangover_frames

        self._buffer: List[np.ndarray] = []
        self._segment_start: Optional[float] = None
        self._silence_start: Optional[float] = None
        self._speech_hangover = 0

    def _frame_energy(self, frame: np.ndarray) -> float:
        float_frame = frame.astype(np.float32) / 32_768.0
        if float_frame.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(float_frame ** 2)))

    def reset(self) -> None:
        """Reset the internal state and discard the buffered audio."""

        self._buffer.clear()
        self._segment_start = None
        self._silence_start = None
        self._speech_hangover = 0

    def process_frame(self, frame: np.ndarray, *, timestamp: float) -> Optional[AudioSegment]:
        """Process a single frame.

        Parameters
        ----------
        frame:
            Audio frame containing ``int16`` samples.
        timestamp:
            Timestamp (seconds) indicating when the frame started.

        Returns
        -------
        AudioSegment or None
            ``AudioSegment`` when a segment has completed, otherwise ``None``.
        """

        if frame.ndim != 1:
            raise ValueError("Audio frames must be one-dimensional.")
        if frame.dtype != np.int16:
            frame = frame.astype(np.int16)
        if frame.size != self.frame_size:
            raise ValueError(f"Expected frame with {self.frame_size} samples, got {frame.size}")

        energy = self._frame_energy(frame)
        is_speech = energy >= self.energy_threshold

        if is_speech:
            if self._segment_start is None:
                self._segment_start = timestamp
            self._buffer.append(frame)
            self._silence_start = None
            self._speech_hangover = self.hangover_frames
            return None

        # silence frame
        if self._segment_start is None:
            return None  # still waiting for speech

        if self._speech_hangover > 0:
            self._speech_hangover -= 1
            self._buffer.append(frame)
            return None

        if self._silence_start is None:
            self._silence_start = timestamp
            return None

        elapsed = timestamp - self._silence_start
        if elapsed < self.silence_duration:
            return None

        segment = self._build_segment(speech_end_time=self._silence_start)
        self.reset()
        return segment

    def flush(self, *, timestamp: float) -> Optional[AudioSegment]:
        """Force the detector to flush any buffered speech as a segment."""

        if self._segment_start is None or not self._buffer:
            self.reset()
            return None

        speech_end = self._silence_start if self._silence_start is not None else timestamp
        segment = self._build_segment(speech_end_time=speech_end)
        self.reset()
        return segment

    def _build_segment(self, speech_end_time: float) -> AudioSegment:
        samples = np.concatenate(self._buffer) if self._buffer else np.array([], dtype=np.int16)
        start_time = self._segment_start or speech_end_time
        return AudioSegment(samples=samples, sample_rate=self.sample_rate, start_time=start_time, end_time=speech_end_time)


class MicrophoneBuffer:
    """High-level buffer that receives microphone frames and emits segments.

    The class wraps :class:`EnergyVADSegmenter` and invokes ``on_segment_ready`` each
    time a speech segment is detected.  Frames can be provided as raw bytes or numpy
    arrays of ``int16`` samples.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        frame_duration: float = 0.03,
        energy_threshold: float = 0.02,
        silence_duration: float = 0.5,
        hangover_frames: int = 3,
        on_segment_ready: Optional[Callable[[AudioSegment], None]] = None,
    ) -> None:
        self.sample_rate = sample_rate
        frame_size = int(sample_rate * frame_duration)
        frame_size -= frame_size % 2  # ensure even number of samples for byte conversion
        if frame_size <= 0:
            raise ValueError("Frame duration is too small for the given sample rate.")

        self.segmenter = EnergyVADSegmenter(
            sample_rate=sample_rate,
            frame_size=frame_size,
            energy_threshold=energy_threshold,
            silence_duration=silence_duration,
            hangover_frames=hangover_frames,
        )
        self.on_segment_ready = on_segment_ready

    def add_audio_frame(self, frame: Sequence[int] | bytes, *, timestamp: Optional[float] = None) -> Optional[AudioSegment]:
        """Add a frame of audio data to the buffer.

        Parameters
        ----------
        frame:
            Either a one-dimensional iterable of integers or raw PCM bytes representing
            ``int16`` samples.
        timestamp:
            Timestamp (seconds) at which the frame starts.  When omitted the current time
            is used.
        """

        if timestamp is None:
            timestamp = time.time()

        np_frame = self._ensure_array(frame)
        frame_size = self.segmenter.frame_size
        last_segment: Optional[AudioSegment] = None
        for offset in range(0, np_frame.size, frame_size):
            chunk = np_frame[offset : offset + frame_size]
            if chunk.size == 0:
                continue
            if chunk.size < frame_size:
                chunk = np.pad(chunk, (0, frame_size - chunk.size), mode="constant")
            chunk_timestamp = timestamp + offset / self.sample_rate
            segment = self.segmenter.process_frame(chunk, timestamp=chunk_timestamp)
            if segment:
                last_segment = segment
                if self.on_segment_ready:
                    self.on_segment_ready(segment)
        return last_segment

    def flush(self, *, timestamp: Optional[float] = None) -> Optional[AudioSegment]:
        """Force any buffered speech to be emitted as a segment."""

        if timestamp is None:
            timestamp = time.time()

        segment = self.segmenter.flush(timestamp=timestamp)
        if segment and self.on_segment_ready:
            self.on_segment_ready(segment)
        return segment

    def _ensure_array(self, frame: Sequence[int] | bytes) -> np.ndarray:
        if isinstance(frame, (bytes, bytearray, memoryview)):
            return np.frombuffer(frame, dtype=np.int16)

        np_frame = np.asarray(frame, dtype=np.int16)
        if np_frame.ndim != 1:
            raise ValueError("Audio frames must be one-dimensional sequences.")
        return np_frame
