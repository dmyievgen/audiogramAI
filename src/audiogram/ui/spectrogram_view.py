"""Spectrogram view: pyqtgraph image + note ticks + playhead + hover info."""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from ..core.models import Spectrogram
from ..core.notes import midi_to_name
from ..core.temperament import Temperament


class _NoteAxis(pg.AxisItem):
    """Y-axis whose values are MIDI numbers but rendered as note names.

    pyqtgraph picks the tick density on its own based on the zoom level —
    we just translate numeric ticks that happen to fall on whole semitones
    into ``C4`` / ``F#3`` style labels.
    """

    def tickStrings(self, values, scale, spacing):  # noqa: N802 (pyqtgraph API)
        labels: list[str] = []
        for value in values:
            nearest = round(float(value))
            if abs(value - nearest) < 0.05 and 0 <= nearest <= 127:
                labels.append(midi_to_name(int(nearest)))
            else:
                labels.append("")
        return labels

    def tickSpacing(self, minVal, maxVal, size):  # noqa: N802 (pyqtgraph API)
        # Force a semitone-aligned grid: major every octave, minor every semitone.
        return [(12.0, 0.0), (1.0, 0.0)]


class _NavViewBox(pg.ViewBox):
    """ViewBox where wheel/two-finger scroll pans, modifiers fall back to zoom.

    - plain wheel / two-finger trackpad scroll → pan along the wheel axis
    - ⌘ (Meta) or Ctrl + wheel → zoom X
    - ⇧ + wheel → zoom Y
    - right-button drag (pyqtgraph default) → rectangular zoom
    """

    def wheelEvent(self, ev, axis=None):  # noqa: N802 (pyqtgraph API)
        mods = ev.modifiers()
        if mods & (
            QtCore.Qt.KeyboardModifier.ControlModifier
            | QtCore.Qt.KeyboardModifier.MetaModifier
        ):
            super().wheelEvent(ev, axis=0)
            return
        if mods & QtCore.Qt.KeyboardModifier.ShiftModifier:
            super().wheelEvent(ev, axis=1)
            return

        delta = ev.delta()
        if delta == 0:
            ev.accept()
            return

        (xmin, xmax), (ymin, ymax) = self.viewRange()
        # 120 units = one classic notch; trackpads send small fractions thereof.
        fraction = -delta / 120.0 * 0.08
        if ev.orientation() == QtCore.Qt.Orientation.Horizontal:
            self.translateBy(x=fraction * (xmax - xmin), y=0)
        else:
            # Natural-scroll feel: drag content with the fingers on the Y axis.
            self.translateBy(x=0, y=-fraction * (ymax - ymin))
        ev.accept()


class _GesturePlotWidget(pg.PlotWidget):
    """PlotWidget that turns macOS pinch gestures into ViewBox zoom."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.grabGesture(QtCore.Qt.GestureType.PinchGesture)

    def event(self, ev) -> bool:  # noqa: D401 — Qt override
        if ev.type() == QtCore.QEvent.Type.Gesture:
            pinch = ev.gesture(QtCore.Qt.GestureType.PinchGesture)
            if pinch is not None and self._handle_pinch(pinch):
                return True
        return super().event(ev)

    def _handle_pinch(self, pinch) -> bool:
        scale = float(pinch.scaleFactor())
        if scale <= 0.0 or scale == 1.0:
            return True
        view_box = self.getPlotItem().getViewBox()
        widget_pt = self.mapFromGlobal(pinch.hotSpot().toPoint())
        scene_pt = self.mapToScene(widget_pt)
        view_pt = view_box.mapSceneToView(scene_pt)
        # scaleFactor > 1 means fingers spread → zoom in (scaleBy with <1).
        view_box.scaleBy((1.0 / scale, 1.0 / scale), center=view_pt)
        return True


class SpectrogramView(QtWidgets.QWidget):
    """Plots a spectrogram, a draggable playhead and a hover crosshair.

    Emits :pyattr:`seek_requested` when the user clicks on the plot.
    """

    seek_requested = QtCore.pyqtSignal(float)
    selection_changed = QtCore.pyqtSignal(float, float)
    selection_cleared = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        pg.setConfigOptions(antialias=False, useOpenGL=False)
        self._note_axis = _NoteAxis(orientation="left")
        self._view_box = _NavViewBox()
        self._plot_widget = _GesturePlotWidget(
            viewBox=self._view_box, axisItems={"left": self._note_axis}
        )
        self._plot = self._plot_widget.getPlotItem()
        self._plot.setLabel("bottom", "Час", units="с")
        self._plot.setLabel("left", "Нота")
        self._plot.showGrid(x=True, y=True, alpha=0.18)
        self._plot.setMouseEnabled(x=True, y=True)

        self._image = pg.ImageItem(axisOrder="row-major")
        self._image.setLookupTable(self._magma_lut())
        self._plot.addItem(self._image)

        self._playhead = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#d7ff64", width=2)
        )
        self._plot.addItem(self._playhead)
        self._playhead.setVisible(False)

        self._selection_region = pg.LinearRegionItem(
            values=(0.0, 0.0),
            orientation="vertical",
            brush=pg.mkBrush(215, 255, 100, 50),
            pen=pg.mkPen("#d7ff64", width=1),
            hoverBrush=pg.mkBrush(215, 255, 100, 80),
            movable=True,
        )
        self._selection_region.setZValue(20)
        self._plot.addItem(self._selection_region, ignoreBounds=True)
        self._selection_region.setVisible(False)
        self._selection_region.sigRegionChanged.connect(self._on_region_user_change)
        self._has_selection = False
        self._suppress_region_signal = False
        self._cursor_seconds = 0.0

        crosshair_pen = pg.mkPen("#9aa", width=1, style=QtCore.Qt.PenStyle.DashLine)
        self._hover_v = pg.InfiniteLine(angle=90, pen=crosshair_pen)
        self._hover_h = pg.InfiniteLine(angle=0, pen=crosshair_pen)
        self._plot.addItem(self._hover_v, ignoreBounds=True)
        self._plot.addItem(self._hover_h, ignoreBounds=True)
        self._hover_v.setVisible(False)
        self._hover_h.setVisible(False)

        # Floating note label that follows the cursor (anchored to its
        # bottom-right corner so the text appears above-and-left of the
        # mouse pointer, never under the finger).
        self._cursor_note = pg.TextItem(
            anchor=(1.0, 1.0),
            color="#d7ff64",
            fill=pg.mkBrush(16, 21, 18, 220),
            border=pg.mkPen("#d7ff64", width=1),
        )
        font = self._cursor_note.textItem.font()
        font.setPointSize(11)
        font.setBold(True)
        self._cursor_note.setFont(font)
        self._cursor_note.setZValue(30)
        self._plot.addItem(self._cursor_note, ignoreBounds=True)
        self._cursor_note.setVisible(False)

        self._zoom_hint = QtWidgets.QLabel(
            "Клік — курсор · ⇧ + клік — виділення · скрол 2 пальцями — пан"
            " · пінч — зум · ⌘/⇧ + скрол — зум X/Y · права кнопка — обласний зум"
        )
        self._zoom_hint.setStyleSheet(
            "color: #6f7a6a; padding: 4px 10px; background: #101512; font-size: 11px;"
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._plot_widget, 1)
        layout.addWidget(self._zoom_hint)

        self._duration = 0.0
        self._lowest_midi = 0
        self._highest_midi = 0
        self._bins_per_semitone = 1
        self._y_start = 0.0
        self._y_end = 0.0
        self._label_transpose = 0.0
        self._temperament = Temperament()

        scene = self._plot_widget.scene()
        self._mouse_proxy = pg.SignalProxy(
            scene.sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )
        scene.sigMouseClicked.connect(self._on_mouse_clicked)

    # ----------------------------------------------------------------- public
    def show_spectrogram(self, spec: Spectrogram, *, preserve_view: bool = False) -> None:
        """Render ``spec``. With ``preserve_view`` keep zoom, selection, playhead.

        Used when the user toggles a *display-only* transform (e.g. harmonic
        suppression) and we just need to swap the underlying image.
        """
        if spec.times.size == 0:
            self.clear()
            return

        data = spec.magnitudes_db
        bps = max(1, spec.bins_per_semitone)
        self._bins_per_semitone = bps
        self._lowest_midi = spec.lowest_midi
        self._highest_midi = spec.highest_midi
        self._duration = float(spec.times[-1])

        # Each row of ``data`` has its CENTER at MIDI ``lowest + k / bps``.
        # Shift the image rect by half a bin so labels align with note centers.
        half_bin = 0.5 / bps
        self._y_start = float(self._lowest_midi) - half_bin
        self._y_end = self._y_start + data.shape[0] / bps

        self._image.setImage(
            data,
            autoLevels=False,
            levels=(float(np.min(data)), float(np.max(data))),
        )
        self._image.setRect(
            QtCore.QRectF(
                0.0, self._y_start, self._duration, self._y_end - self._y_start
            )
        )

        if preserve_view:
            return

        view_box = self._plot.getViewBox()
        view_box.setLimits(
            xMin=0.0,
            xMax=self._duration,
            yMin=self._y_start,
            yMax=self._y_end,
            minXRange=0.05,
            minYRange=2.0,
        )
        self._plot.setXRange(0.0, self._duration, padding=0)
        self._plot.setYRange(self._y_start, self._y_end, padding=0)

        self._selection_region.setBounds([0.0, self._duration])

        self._playhead.setPos(0.0)
        self._playhead.setVisible(True)
        self._cursor_seconds = 0.0
        self.clear_selection()

    def clear(self) -> None:
        self._image.clear()
        self._playhead.setVisible(False)
        self._hover_v.setVisible(False)
        self._hover_h.setVisible(False)
        self._cursor_note.setVisible(False)
        self._duration = 0.0
        self.clear_selection()

    def update_playhead(self, seconds: float) -> None:
        if self._duration <= 0.0:
            return
        bounded = max(0.0, min(seconds, self._duration))
        self._playhead.setPos(bounded)
        self._cursor_seconds = bounded

    # ------------------------------------------------------------------- zoom
    def zoom(self, factor: float, axis: str = "both") -> None:
        """Multiply the visible range by ``factor`` (``<1`` zooms in)."""
        if self._duration <= 0.0:
            return
        view_box = self._plot.getViewBox()
        x = factor if axis in ("x", "both") else 1.0
        y = factor if axis in ("y", "both") else 1.0
        view_box.scaleBy((x, y))

    def fit_view(self) -> None:
        if self._duration <= 0.0:
            return
        self._plot.setXRange(0.0, self._duration, padding=0)
        self._plot.setYRange(self._y_start, self._y_end, padding=0)

    # ------------------------------------------------------------- transpose
    def set_label_transpose(self, semitones: float) -> None:
        """Shift the *displayed* note name / frequency by ``semitones``.

        The underlying spectrogram data and Y-axis ticks stay untouched —
        only the hover read-out is affected.
        """
        self._label_transpose = float(semitones)

    def set_temperament(self, temperament: Temperament) -> None:
        """Apply a tuning system to the cursor read-out only."""
        self._temperament = temperament

    # -------------------------------------------------------------- selection
    def has_selection(self) -> bool:
        return self._has_selection

    def selection_range(self) -> tuple[float, float] | None:
        if not self._has_selection:
            return None
        lo, hi = self._selection_region.getRegion()
        return float(lo), float(hi)

    def clear_selection(self) -> None:
        if not self._has_selection:
            self._selection_region.setVisible(False)
            return
        self._has_selection = False
        self._selection_region.setVisible(False)
        self.selection_cleared.emit()

    def _set_region_silently(self, lo: float, hi: float) -> None:
        self._suppress_region_signal = True
        try:
            self._selection_region.setRegion((lo, hi))
        finally:
            self._suppress_region_signal = False

    def _on_region_user_change(self) -> None:
        if self._suppress_region_signal or not self._has_selection:
            return
        lo, hi = self._selection_region.getRegion()
        lo = float(max(0.0, min(self._duration, lo)))
        hi = float(max(0.0, min(self._duration, hi)))
        if hi - lo < 1e-4:
            return
        self.selection_changed.emit(lo, hi)

    # ---------------------------------------------------------------- helpers
    def _on_mouse_clicked(self, event) -> None:
        if self._duration <= 0.0:
            return
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        view_pos = self._plot.vb.mapSceneToView(event.scenePos())
        seconds = float(view_pos.x())
        if not (0.0 <= seconds <= self._duration):
            return

        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            anchor = self._cursor_seconds
            lo, hi = (anchor, seconds) if anchor <= seconds else (seconds, anchor)
            if hi - lo < 1e-4:
                return
            event.accept()
            self._set_region_silently(lo, hi)
            self._selection_region.setVisible(True)
            self._has_selection = True
            self.selection_changed.emit(lo, hi)
            return

        self._cursor_seconds = seconds
        self.clear_selection()
        self.seek_requested.emit(seconds)

    def _on_mouse_moved(self, args) -> None:
        if self._duration <= 0.0:
            return

        scene_pos = args[0]
        view_box = self._plot.vb
        if not view_box.sceneBoundingRect().contains(scene_pos):
            self._hover_v.setVisible(False)
            self._hover_h.setVisible(False)
            self._cursor_note.setVisible(False)
            return

        view_pos = view_box.mapSceneToView(scene_pos)
        time_seconds = float(view_pos.x())
        midi_value = float(view_pos.y())

        if not (0.0 <= time_seconds <= self._duration):
            return
        if not (self._y_start <= midi_value <= self._y_end):
            return

        # Apply temperament FIRST: snap raw cursor position to the nearest
        # note in the active tuning system, then shift that label by the
        # user-configured transposition.
        base_midi, _ = self._temperament.nearest_note(midi_value)
        display_midi = max(0, min(127, base_midi + int(round(self._label_transpose))))
        frequency = 440.0 * math.pow(
            2.0, (midi_value + self._label_transpose - 69) / 12.0
        )
        note_label = midi_to_name(display_midi)

        self._hover_v.setPos(time_seconds)
        self._hover_h.setPos(midi_value)
        self._hover_v.setVisible(True)
        self._hover_h.setVisible(True)

        # Floating cursor label: place it ABOVE-AND-LEFT of the pointer
        # with a small pixel offset so it stays out of the cursor's way.
        offset_px = QtCore.QPointF(-10.0, -10.0)
        anchor_scene = scene_pos + offset_px
        anchor_view = view_box.mapSceneToView(anchor_scene)
        self._cursor_note.setPos(anchor_view.x(), anchor_view.y())
        self._cursor_note.setText(
            f"{note_label}   {frequency:.1f} Гц"
        )
        self._cursor_note.setVisible(True)

    @staticmethod
    def _magma_lut() -> np.ndarray:
        # Approximation of the magma colormap without pulling in matplotlib.
        stops = np.array(
            [
                [0, 0, 4],
                [28, 16, 68],
                [79, 18, 123],
                [129, 37, 129],
                [181, 54, 122],
                [229, 80, 100],
                [251, 135, 97],
                [254, 194, 135],
                [252, 253, 191],
            ],
            dtype=np.float32,
        )
        positions = np.linspace(0.0, 1.0, stops.shape[0])
        xs = np.linspace(0.0, 1.0, 256)
        lut = np.empty((256, 3), dtype=np.uint8)
        for channel in range(3):
            lut[:, channel] = np.interp(xs, positions, stops[:, channel]).astype(
                np.uint8
            )
        return lut
