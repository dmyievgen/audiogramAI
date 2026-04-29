"""Pure-data structures shared across layers. No Qt, no audio backends here."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AudioTrack:
    """Decoded mono audio buffer ready for analysis and playback."""

    path: Path
    samples: np.ndarray  # float32, shape (n_samples,)
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return float(self.samples.shape[0]) / float(self.sample_rate)


@dataclass(frozen=True)
class Spectrogram:
    """Magnitude spectrogram aligned to a fixed semitone grid.

    Y axis is expressed in MIDI units. Row ``k`` corresponds to MIDI value
    ``lowest_midi + k / bins_per_semitone``, so increasing ``bins_per_semitone``
    just makes the rows thinner without changing the pitch range.
    """

    magnitudes_db: np.ndarray  # shape (n_bins, n_frames)
    times: np.ndarray  # shape (n_frames,) — seconds for each column
    lowest_midi: int
    bins_per_semitone: int

    @property
    def n_bins(self) -> int:
        return int(self.magnitudes_db.shape[0])

    @property
    def highest_midi(self) -> int:
        """Highest *integer* MIDI value covered by at least one bin."""
        return self.lowest_midi + (self.n_bins - 1) // self.bins_per_semitone
