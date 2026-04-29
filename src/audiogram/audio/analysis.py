"""Compute a constant-Q spectrogram aligned to MIDI semitone bins."""
from __future__ import annotations

import math

import librosa
import numpy as np
from scipy.ndimage import maximum_filter

from ..core.models import AudioTrack, Spectrogram

# Range covered by the Y-axis. C1 == MIDI 24, B7 == MIDI 95 → 72 semitones.
DEFAULT_LOWEST_MIDI = 24
DEFAULT_OCTAVES = 6
DEFAULT_BINS_PER_SEMITONE = 3
DEFAULT_HOP_LENGTH = 1024


def compute_spectrogram(
    track: AudioTrack,
    *,
    lowest_midi: int = DEFAULT_LOWEST_MIDI,
    octaves: int = DEFAULT_OCTAVES,
    bins_per_semitone: int = DEFAULT_BINS_PER_SEMITONE,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> Spectrogram:
    """Run a CQT and return a :class:`Spectrogram` with note-aligned bins."""
    n_bins = octaves * 12 * bins_per_semitone
    fmin = librosa.midi_to_hz(lowest_midi)

    cqt = librosa.cqt(
        y=track.samples,
        sr=track.sample_rate,
        hop_length=hop_length,
        fmin=float(fmin),
        n_bins=n_bins,
        bins_per_octave=12 * bins_per_semitone,
    )
    magnitudes_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    n_frames = magnitudes_db.shape[1]
    times = librosa.frames_to_time(
        np.arange(n_frames),
        sr=track.sample_rate,
        hop_length=hop_length,
    )

    return Spectrogram(
        magnitudes_db=magnitudes_db.astype(np.float32, copy=False),
        times=times.astype(np.float32, copy=False),
        lowest_midi=int(lowest_midi),
        bins_per_semitone=int(bins_per_semitone),
    )


def suppress_harmonics(
    spec: Spectrogram,
    *,
    n_harmonics: int = 6,
    neighborhood_semitones: float = 1.0,
    rel_threshold_db: float = -28.0,
    floor_db: float = -80.0,
) -> Spectrogram:
    """Collapse harmonic stacks to one peak per perceived note.

    Strategy:

    1. **Harmonic salience.** For every bin ``k`` we sum the linear
       magnitude at ``k`` plus the bins where its 2nd…N-th harmonics
       *would land*. A bin that is a true fundamental gets boosted by
       its overtones; a bin that is itself an overtone of something
       lower gets no extra evidence and stays small relative to its
       parent.
    2. **Non-maximum suppression.** Within a ±``neighborhood_semitones``
       window along the pitch axis we keep only local maxima of the
       salience — this removes the smear left by CQT bin width.
    3. **Threshold.** A peak survives only if it is within
       ``rel_threshold_db`` of the loudest peak in its frame.

    The output spectrogram has the original magnitude at surviving
    peaks and a flat ``floor_db`` background everywhere else, so the
    image looks like a constellation of single dots, one per note.
    """
    bps = max(1, spec.bins_per_semitone)
    db = spec.magnitudes_db
    magnitude = np.power(10.0, db / 20.0)
    n_bins, n_frames = magnitude.shape
    if n_bins == 0 or n_frames == 0:
        return spec

    salience = magnitude.copy()
    for h in range(2, max(2, n_harmonics) + 1):
        delta = int(round(12.0 * math.log2(h) * bps))
        if delta <= 0 or delta >= n_bins:
            break
        # Magnitude of bin (k + delta) is the H-th harmonic of bin k,
        # so it contributes to the salience of bin k.
        shifted = np.zeros_like(magnitude)
        shifted[: n_bins - delta, :] = magnitude[delta:, :]
        salience += shifted

    win = max(1, int(round(neighborhood_semitones * bps)))
    local_max = maximum_filter(salience, size=(2 * win + 1, 1))
    is_peak = (salience >= local_max) & (salience > 0)

    # Per-frame relative threshold: ignore peaks far below the frame's loudest.
    frame_max = salience.max(axis=0, keepdims=True) + 1e-12
    rel_db = 20.0 * np.log10((salience + 1e-12) / frame_max)
    is_peak &= rel_db >= rel_threshold_db

    floor_lin = 10.0 ** (floor_db / 20.0)
    out = np.full_like(magnitude, floor_lin)
    out[is_peak] = magnitude[is_peak]

    ref = float(np.max(out)) or 1.0
    new_db = 20.0 * np.log10(out / ref)

    return Spectrogram(
        magnitudes_db=new_db.astype(np.float32, copy=False),
        times=spec.times,
        lowest_midi=spec.lowest_midi,
        bins_per_semitone=spec.bins_per_semitone,
    )
