"""Utilities for converting between MIDI numbers, frequencies and note names."""
from __future__ import annotations

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_to_name(midi: int) -> str:
    """Return note label like ``C1`` or ``F#3`` for a given MIDI number."""
    octave = midi // 12 - 1
    name = NOTE_NAMES[midi % 12]
    return f"{name}{octave}"


def name_to_midi(name: str) -> int:
    """Inverse of :func:`midi_to_name`. Accepts ``C#3`` style labels."""
    if len(name) < 2:
        raise ValueError(f"Invalid note name: {name!r}")

    if name[1] == "#":
        pitch_class = NOTE_NAMES.index(name[:2])
        octave = int(name[2:])
    else:
        pitch_class = NOTE_NAMES.index(name[0])
        octave = int(name[1:])

    return (octave + 1) * 12 + pitch_class
