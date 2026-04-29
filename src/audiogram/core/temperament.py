"""Tuning systems (temperaments) for note labels and pitch detection.

The application's spectrogram is computed in equal-tempered semitone bins
(CQT). The temperament objects defined here only control *how note labels
are placed and named* on top of that data: in equal temperament the labels
sit at integer MIDI positions, while in just intonation they shift by a
small number of cents relative to the chosen tonic frequency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# 5-limit just intonation ratios for each interval (in semitones) above the tonic.
JUST_RATIOS: tuple[float, ...] = (
    1.0 / 1.0,    # unison
    16.0 / 15.0,  # minor second
    9.0 / 8.0,    # major second
    6.0 / 5.0,    # minor third
    5.0 / 4.0,    # major third
    4.0 / 3.0,    # perfect fourth
    45.0 / 32.0,  # tritone
    3.0 / 2.0,    # perfect fifth
    8.0 / 5.0,    # minor sixth
    5.0 / 3.0,    # major sixth
    9.0 / 5.0,    # minor seventh
    15.0 / 8.0,   # major seventh
)

_JUST_SEMITONES: tuple[float, ...] = tuple(
    12.0 * math.log2(r) for r in JUST_RATIOS
)


def _freq_to_midi(freq: float) -> float:
    return 69.0 + 12.0 * math.log2(max(1e-6, freq) / 440.0)


@dataclass(frozen=True)
class Temperament:
    """Description of a tuning system used purely for label placement."""

    kind: str = "equal"        # "equal" | "just"
    tonic_freq: float = 440.0  # tonic pitch in Hz (default: A4 = 440 Hz)

    @property
    def tonic_midi(self) -> float:
        """Tonic position on the (float) MIDI scale."""
        return _freq_to_midi(self.tonic_freq)

    @property
    def tonic_pc(self) -> int:
        """Pitch class (0..11) the tonic frequency is nearest to."""
        return int(round(self.tonic_midi)) % 12

    def label_position(self, midi: int) -> float:
        """Y-axis position (in equal-temperament semitone units) of MIDI ``midi``."""
        if self.kind != "just":
            return float(midi)
        tonic_midi = self.tonic_midi
        tonic_int = int(round(tonic_midi))
        tonic_pc = tonic_int % 12
        interval = (midi - tonic_pc) % 12
        # nearest same-pitch-class tonic instance at or below ``midi``
        nearest_tonic_int = midi - interval
        # pin that octave's tonic to the user's exact frequency
        tonic_offset = tonic_midi - tonic_int
        return nearest_tonic_int + tonic_offset + _JUST_SEMITONES[interval]

    def nearest_note(self, midi_value: float) -> tuple[int, float]:
        """Find the closest temperament-aligned note to ``midi_value``.

        Returns ``(midi_number, label_position)`` — the integer MIDI number of
        the nearest note and the Y position where its label sits.
        """
        if self.kind != "just":
            m = max(0, min(127, int(round(midi_value))))
            return m, float(m)

        best_midi = 0
        best_pos = 0.0
        best_dist = math.inf
        center = int(round(midi_value))
        for midi in range(max(0, center - 12), min(127, center + 12) + 1):
            pos = self.label_position(midi)
            dist = abs(pos - midi_value)
            if dist < best_dist:
                best_dist = dist
                best_midi = midi
                best_pos = pos
        return best_midi, best_pos
