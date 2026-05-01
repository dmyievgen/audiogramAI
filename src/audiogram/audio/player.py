"""Threaded audio playback with frame-accurate position tracking.

The class is intentionally minimal — load a track, ``play()`` / ``pause()`` /
``stop()`` / ``seek()``, and read the current playback time. The UI polls the
position with a ``QTimer`` so this module stays Qt-free.
"""
from __future__ import annotations

import threading
from typing import Any, Optional

import numpy as np
import sounddevice as sd

from ..core.models import AudioTrack


class AudioPlayer:
    def __init__(self) -> None:
        self._track: Optional[AudioTrack] = None
        self._stream: Optional[sd.OutputStream] = None
        self._frame_index = 0
        self._lock = threading.Lock()
        self._loop_enabled = False
        self._play_region_enabled = False
        self._loop_start_frame = 0
        self._loop_end_frame = 0
        self._playback_rate = 1.0
        self._output_device_index: Optional[int] = None
        self._output_device_signature: Optional[tuple[Any, ...]] = None

    # ------------------------------------------------------------------- loop
    def set_loop_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._loop_enabled = bool(enabled)

    def set_play_region_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._play_region_enabled = bool(enabled)

    def set_loop_region(self, start_seconds: Optional[float], end_seconds: Optional[float]) -> None:
        with self._lock:
            if (
                self._track is None
                or start_seconds is None
                or end_seconds is None
                or end_seconds <= start_seconds
            ):
                self._loop_start_frame = 0
                self._loop_end_frame = 0
                return
            sr = self._track.sample_rate
            total = self._track.samples.shape[0]
            start = int(max(0.0, start_seconds) * sr)
            end = int(max(0.0, end_seconds) * sr)
            self._loop_start_frame = max(0, min(start, total))
            self._loop_end_frame = max(self._loop_start_frame, min(end, total))

    # ------------------------------------------------------------------ track
    def load(self, track: AudioTrack) -> None:
        self.stop()
        with self._lock:
            self._track = track
            self._frame_index = 0
            self._play_region_enabled = False
            self._loop_start_frame = 0
            self._loop_end_frame = 0

    @property
    def track(self) -> Optional[AudioTrack]:
        return self._track

    # ------------------------------------------------------------------ state
    @property
    def is_playing(self) -> bool:
        try:
            return self._stream is not None and self._stream.active
        except Exception:
            return False

    def position_seconds(self) -> float:
        with self._lock:
            if self._track is None:
                return 0.0
            return self._frame_index / float(self._track.sample_rate)

    # --------------------------------------------------------------- controls
    def play(self) -> None:
        if self._track is None or self.is_playing:
            return

        self._refresh_default_output()
        stream_kwargs: dict[str, object] = {
            "samplerate": self._effective_samplerate(),
            "channels": 1,
            "dtype": "float32",
            "callback": self._callback,
        }
        if self._output_device_index is not None:
            stream_kwargs["device"] = self._output_device_index
        self._stream = sd.OutputStream(**stream_kwargs)
        self._stream.start()

    def set_playback_rate(self, rate: float) -> None:
        """Varispeed factor (1.0 = native). Both tempo *and* pitch change.

        Implemented by pushing samples through the sound device at
        ``track.sample_rate * rate`` while ``frame_index`` advances at
        the original rate — so :meth:`position_seconds` keeps reporting
        time on the **original** timeline.
        """
        rate = float(max(0.05, min(8.0, rate)))
        was_playing = self.is_playing
        with self._lock:
            self._playback_rate = rate
        if was_playing:
            # Restart the stream at the new sample rate, keeping position.
            self._teardown_stream()
            self.play()

    @property
    def playback_rate(self) -> float:
        return self._playback_rate

    def _effective_samplerate(self) -> int:
        assert self._track is not None
        return max(1, int(round(self._track.sample_rate * self._playback_rate)))

    def pause(self) -> None:
        self._teardown_stream()

    def stop(self) -> None:
        self._teardown_stream()
        with self._lock:
            self._frame_index = 0

    def seek(self, seconds: float) -> None:
        with self._lock:
            if self._track is None:
                return
            total = self._track.samples.shape[0]
            target = int(max(0.0, seconds) * self._track.sample_rate)
            self._frame_index = min(target, total)

    def refresh_output_device(self) -> bool:
        """Switch playback to the current system default output if it changed."""
        if not self._refresh_default_output():
            return False

        if self._stream is not None:
            self._teardown_stream()
            self.play()
        return True

    # --------------------------------------------------------------- internal
    def _refresh_default_output(self) -> bool:
        index = self._current_default_output_index()
        signature = self._device_signature(index)
        if signature == self._output_device_signature:
            return False
        self._output_device_index = index
        self._output_device_signature = signature
        return True

    def _current_default_output_index(self) -> Optional[int]:
        try:
            device = sd.default.device
            if len(device) >= 2 and isinstance(device[1], int) and device[1] >= 0:
                return int(device[1])
        except Exception:
            pass

        try:
            hostapi = int(sd.default.hostapi)
            hostapis = sd.query_hostapis()
            if 0 <= hostapi < len(hostapis):
                index = int(hostapis[hostapi].get("default_output_device", -1))
                if index >= 0:
                    return index
            for api in hostapis:
                index = int(api.get("default_output_device", -1))
                if index >= 0:
                    return index
        except Exception:
            pass

        return None

    def _device_signature(self, index: Optional[int]) -> Optional[tuple[Any, ...]]:
        if index is None:
            return None
        try:
            info = sd.query_devices(index)
        except Exception:
            return (index,)
        return (
            index,
            info.get("name"),
            info.get("hostapi"),
            info.get("max_output_channels"),
        )

    def _teardown_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def _callback(self, outdata, frames, _time, _status) -> None:  # pragma: no cover
        with self._lock:
            track = self._track
            if track is None:
                outdata.fill(0)
                raise sd.CallbackStop

            samples = track.samples
            total = samples.shape[0]
            bounded_region = self._loop_end_frame > self._loop_start_frame
            loop = (
                self._loop_enabled
                and bounded_region
            )
            play_region = self._play_region_enabled and bounded_region

            idx = self._frame_index
            if (loop or play_region) and (
                idx < self._loop_start_frame or idx >= self._loop_end_frame
            ):
                idx = self._loop_start_frame

            written = 0
            stop_after = False
            while written < frames:
                end_limit = self._loop_end_frame if (loop or play_region) else total
                available = end_limit - idx
                if available <= 0:
                    if loop:
                        idx = self._loop_start_frame
                        continue
                    stop_after = True
                    break
                n = min(frames - written, available)
                outdata[written : written + n, 0] = samples[idx : idx + n]
                idx += n
                written += n
                if loop and idx >= end_limit:
                    idx = self._loop_start_frame

            self._frame_index = idx

        if written < frames:
            outdata[written:, 0] = 0.0
        if stop_after:
            raise sd.CallbackStop
