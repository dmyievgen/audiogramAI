"""Load audio files into a normalized mono float32 buffer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from ..core.models import AudioTrack

SUPPORTED_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a"}


def load_track(path: str | Path) -> AudioTrack:
    """Read an audio file and return a mono float32 :class:`AudioTrack`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32, copy=False)

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 1.0:
        mono = mono / peak

    return AudioTrack(path=path, samples=mono, sample_rate=int(sample_rate))
