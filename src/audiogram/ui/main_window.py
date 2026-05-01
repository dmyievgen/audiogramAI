"""Main window: drop zone, controls, spectrogram."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from ..audio.analysis import compute_spectrogram, suppress_harmonics
from ..audio.loader import SUPPORTED_SUFFIXES, load_track
from ..audio.player import AudioPlayer
from ..core.models import AudioTrack, Spectrogram
from ..core.temperament import Temperament
from ..i18n import I18n
from .spectrogram_view import SpectrogramView


class MainWindow(QtWidgets.QMainWindow):
    PLAYHEAD_INTERVAL_MS = 30
    DEFAULT_SHARPNESS = 50
    DEFAULT_SPEED_PERCENT = 100
    DEFAULT_LABEL_TRANSPOSE = 0.0
    DEFAULT_AUDIO_TRANSPOSE = 0.0
    DEFAULT_TEMPERAMENT_INDEX = 0
    DEFAULT_TONIC_FREQ = 440.0
    DEFAULT_LOOP_ENABLED = True
    DEFAULT_LANGUAGE = "uk"

    def __init__(self) -> None:
        super().__init__()
        self._i18n = I18n(self.DEFAULT_LANGUAGE)
        self._status_key = "ui.status.ready"
        self._status_params: dict[str, object] = {}

        self.setWindowTitle(self._tr("ui.window.title"))
        self.resize(1180, 640)
        self.setAcceptDrops(True)

        self._player = AudioPlayer()

        self._open_action = QtGui.QAction(self._tr("ui.action.open"), self)
        self._open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self._open_action.triggered.connect(self._on_open_dialog)

        self._play_action = QtGui.QAction(self._tr("ui.action.play"), self)
        self._play_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Space))
        self._play_action.setEnabled(False)
        self._play_action.triggered.connect(self._on_toggle_playback)

        self._stop_action = QtGui.QAction(self._tr("ui.action.stop"), self)
        self._stop_action.setEnabled(False)
        self._stop_action.triggered.connect(self._on_stop)

        self._reset_action = QtGui.QAction(self._tr("ui.action.reset"), self)
        self._reset_action.setEnabled(False)
        self._reset_action.triggered.connect(self._on_reset)

        self._fit_action = QtGui.QAction(self._tr("ui.action.fit"), self)
        self._fit_action.setShortcut(QtGui.QKeySequence("Ctrl+0"))
        self._fit_action.setEnabled(False)
        self._fit_action.triggered.connect(lambda: self._spectrogram_view.fit_view())

        self._clear_selection_action = QtGui.QAction(
            self._tr("ui.action.clear_selection"), self
        )
        self._clear_selection_action.setEnabled(False)
        self._clear_selection_action.triggered.connect(
            lambda: self._spectrogram_view.clear_selection()
        )

        self._suppress_harmonics_cb = QtWidgets.QCheckBox(
            self._tr("ui.checkbox.suppress_harmonics")
        )
        self._suppress_harmonics_cb.setToolTip(self._tr("ui.tooltip.suppress_harmonics"))
        self._suppress_harmonics_cb.toggled.connect(self._on_harmonics_toggled)

        self._sharpness_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._sharpness_slider.setRange(0, 100)
        self._sharpness_slider.setValue(50)
        self._sharpness_slider.setFixedWidth(120)
        self._sharpness_slider.setEnabled(False)
        self._sharpness_slider.setToolTip(self._tr("ui.tooltip.sharpness"))
        self._sharpness_slider.valueChanged.connect(self._on_sharpness_changed)
        # Debounce so dragging the slider doesn't recompute on every step.
        self._sharpness_timer = QtCore.QTimer(self)
        self._sharpness_timer.setSingleShot(True)
        self._sharpness_timer.setInterval(150)
        self._sharpness_timer.timeout.connect(self._recompute_suppressed)

        self._speed_spin = QtWidgets.QSpinBox()
        self._speed_spin.setRange(25, 200)
        self._speed_spin.setSingleStep(5)
        self._speed_spin.setValue(100)
        self._speed_spin.setSuffix(self._tr("ui.unit.percent_suffix"))
        self._speed_spin.setToolTip(self._tr("ui.tooltip.speed"))
        self._speed_spin.valueChanged.connect(self._on_speed_changed)

        self._speed_reset_btn = QtWidgets.QToolButton()
        self._speed_reset_btn.setText("↺")
        self._speed_reset_btn.setToolTip(self._tr("ui.tooltip.speed_reset"))
        self._speed_reset_btn.clicked.connect(lambda: self._speed_spin.setValue(100))

        self._loop_cb = QtWidgets.QCheckBox(self._tr("ui.checkbox.loop"))
        self._loop_cb.setChecked(self.DEFAULT_LOOP_ENABLED)
        self._loop_cb.setToolTip(self._tr("ui.tooltip.loop"))
        self._loop_cb.toggled.connect(self._on_loop_toggled)

        self._transpose_indicator = QtWidgets.QDoubleSpinBox()
        self._transpose_indicator.setDecimals(2)
        self._transpose_indicator.setRange(-12.0, 12.0)
        self._transpose_indicator.setSingleStep(1.0)
        self._transpose_indicator.setValue(0.0)
        self._transpose_indicator.setToolTip(self._tr("ui.tooltip.label_transpose"))
        self._transpose_indicator.valueChanged.connect(
            lambda value: self._spectrogram_view.set_label_transpose(value)
        )
        self._transpose_indicator_reset_btn = QtWidgets.QToolButton()
        self._transpose_indicator_reset_btn.setText("↺")
        self._transpose_indicator_reset_btn.setToolTip(
            self._tr("ui.tooltip.label_transpose_reset")
        )
        self._transpose_indicator_reset_btn.clicked.connect(
            lambda: self._transpose_indicator.setValue(0.0)
        )

        self._audio_transpose = QtWidgets.QDoubleSpinBox()
        self._audio_transpose.setDecimals(2)
        self._audio_transpose.setRange(-12.0, 12.0)
        self._audio_transpose.setSingleStep(1.0)
        self._audio_transpose.setValue(0.0)
        self._audio_transpose.setToolTip(self._tr("ui.tooltip.audio_transpose"))
        self._audio_transpose.valueChanged.connect(self._on_audio_transpose_changed)
        self._audio_transpose_timer = QtCore.QTimer(self)
        self._audio_transpose_timer.setSingleShot(True)
        self._audio_transpose_timer.setInterval(250)
        self._audio_transpose_timer.timeout.connect(self._apply_audio_transpose)
        self._audio_transpose_reset_btn = QtWidgets.QToolButton()
        self._audio_transpose_reset_btn.setText("↺")
        self._audio_transpose_reset_btn.setToolTip(
            self._tr("ui.tooltip.audio_transpose_reset")
        )
        self._audio_transpose_reset_btn.clicked.connect(
            lambda: self._audio_transpose.setValue(0.0)
        )

        self._temperament_combo = QtWidgets.QComboBox()
        self._temperament_combo.addItem("", "equal")
        self._temperament_combo.addItem("", "just")
        self._temperament_combo.setToolTip(self._tr("ui.tooltip.temperament"))
        self._temperament_combo.currentIndexChanged.connect(self._on_temperament_changed)

        self._language_combo = QtWidgets.QComboBox()
        self._language_combo.addItem("", "uk")
        self._language_combo.addItem("", "en")
        self._language_combo.currentIndexChanged.connect(self._on_language_changed)
        self._language_combo.setCurrentIndex(0)

        self._tonic_freq = QtWidgets.QDoubleSpinBox()
        self._tonic_freq.setDecimals(1)
        self._tonic_freq.setRange(20.0, 8000.0)
        self._tonic_freq.setSingleStep(0.5)
        self._tonic_freq.setSuffix(self._tr("ui.unit.hz_suffix"))
        self._tonic_freq.setValue(440.0)
        self._tonic_freq.setToolTip(self._tr("ui.tooltip.tonic_freq"))
        self._tonic_freq.valueChanged.connect(self._on_temperament_changed)
        self._tonic_label = QtWidgets.QLabel()

        toolbar = self.addToolBar(self._tr("ui.toolbar.main"))
        self._main_toolbar = toolbar
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        # Hide the overflow chevron: never let actions collapse into a popup.
        toolbar.layout().setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        toolbar.addAction(self._open_action)
        toolbar.addSeparator()
        toolbar.addAction(self._play_action)
        toolbar.addAction(self._stop_action)
        toolbar.addAction(self._reset_action)
        toolbar.addSeparator()
        toolbar.addAction(self._fit_action)
        toolbar.addSeparator()
        toolbar.addAction(self._clear_selection_action)
        toolbar.addSeparator()
        toolbar.addWidget(self._suppress_harmonics_cb)
        self._sharpness_label = QtWidgets.QLabel()
        self._sharpness_label.setEnabled(False)
        toolbar.addWidget(self._sharpness_label)
        toolbar.addWidget(self._sharpness_slider)
        toolbar.addSeparator()
        self._speed_label = QtWidgets.QLabel()
        toolbar.addWidget(self._speed_label)
        toolbar.addWidget(self._speed_spin)
        toolbar.addWidget(self._speed_reset_btn)
        toolbar.addWidget(self._loop_cb)

        # Force a second row so transposition + temperament always fit.
        self.addToolBarBreak()
        toolbar2 = self.addToolBar(self._tr("ui.toolbar.tuning"))
        self._tuning_toolbar = toolbar2
        toolbar2.setMovable(False)
        toolbar2.setFloatable(False)
        toolbar2.layout().setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        self._temperament_label = QtWidgets.QLabel()
        toolbar2.addWidget(self._temperament_label)
        toolbar2.addWidget(self._temperament_combo)
        toolbar2.addWidget(self._tonic_label)
        toolbar2.addWidget(self._tonic_freq)
        toolbar2.addSeparator()
        self._label_transpose_label = QtWidgets.QLabel()
        toolbar2.addWidget(self._label_transpose_label)
        toolbar2.addWidget(self._transpose_indicator)
        toolbar2.addWidget(self._transpose_indicator_reset_btn)
        toolbar2.addSeparator()
        self._audio_transpose_label = QtWidgets.QLabel()
        toolbar2.addWidget(self._audio_transpose_label)
        toolbar2.addWidget(self._audio_transpose)
        toolbar2.addWidget(self._audio_transpose_reset_btn)
        toolbar2.addSeparator()
        self._language_label = QtWidgets.QLabel()
        toolbar2.addWidget(self._language_label)
        toolbar2.addWidget(self._language_combo)

        self._spectrogram_view = SpectrogramView()
        self._spectrogram_view.set_i18n(self._i18n)
        self._spectrogram_view.seek_requested.connect(self._on_seek)
        self._spectrogram_view.playback_seek_requested.connect(
            self._on_playback_seek
        )
        self._spectrogram_view.selection_changed.connect(self._on_selection_changed)
        self._spectrogram_view.selection_cleared.connect(self._on_selection_cleared)
        # Apply initial temperament + tonic-combo visibility.
        self._on_temperament_changed()

        self._drop_hint = QtWidgets.QLabel(self._tr("ui.drop_hint"))
        self._drop_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._drop_hint.setStyleSheet(
            "color: #b4c7a8; font-size: 16px; padding: 24px;"
        )

        self._stack = QtWidgets.QStackedWidget()
        self._stack.addWidget(self._drop_hint)
        self._stack.addWidget(self._spectrogram_view)
        self.setCentralWidget(self._stack)

        self._retranslate_ui()
        self._update_status_text()

        self._playhead_timer = QtCore.QTimer(self)
        self._playhead_timer.setInterval(self.PLAYHEAD_INTERVAL_MS)
        self._playhead_timer.timeout.connect(self._tick)

        self._audio_device_timer = QtCore.QTimer(self)
        self._audio_device_timer.setInterval(1000)
        self._audio_device_timer.timeout.connect(self._refresh_audio_output_device)
        self._audio_device_timer.start()

        self._current_track: Optional[AudioTrack] = None
        self._current_spec: Optional[Spectrogram] = None
        self._suppressed_spec: Optional[Spectrogram] = None
        self._cue_seconds: float = 0.0
        # Cached playback rate the player buffer was built at — used to
        # convert the player's processed-timeline position back to the
        # original timeline when the rate changes.
        self._previous_rate: float = 1.0

    # -------------------------------------------------------------- DnD hooks
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802
        if self._extract_audio_path(event.mimeData().urls()) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802
        path = self._extract_audio_path(event.mimeData().urls())
        if path is not None:
            event.acceptProposedAction()
            self._load_path(path)

    @staticmethod
    def _extract_audio_path(urls: Iterable[QtCore.QUrl]) -> Optional[Path]:
        for url in urls:
            if not url.isLocalFile():
                continue
            candidate = Path(url.toLocalFile())
            if candidate.suffix.lower() in SUPPORTED_SUFFIXES:
                return candidate
        return None

    # ----------------------------------------------------------------- actions
    def _on_open_dialog(self) -> None:
        filters = (
            self._tr("ui.dialog.audio_filter")
            + " ("
            + " ".join(f"*{s}" for s in sorted(SUPPORTED_SUFFIXES))
            + ")"
        )
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, self._tr("ui.dialog.open_audio_title"), "", filters
        )
        if filename:
            self._load_path(Path(filename))

    def _on_toggle_playback(self) -> None:
        if self._player.is_playing:
            self._player.pause()
            self._playhead_timer.stop()
            self._player.seek(self._cue_seconds / self._speed_rate())
            self._spectrogram_view.update_playhead(self._cue_seconds)
            self._spectrogram_view.set_playback_active(False)
            self._play_action.setText(self._tr("ui.action.play"))
        else:
            self._prepare_selection_playback()
            self._player.play()
            if self._player.is_playing:
                self._playhead_timer.start()
                self._spectrogram_view.set_playback_active(True)
                self._play_action.setText(self._tr("ui.action.pause"))

    def _on_stop(self) -> None:
        self._player.stop()
        self._playhead_timer.stop()
        self._player.seek(self._cue_seconds / self._speed_rate())
        self._play_action.setText(self._tr("ui.action.play"))
        self._spectrogram_view.set_playback_active(False)
        self._spectrogram_view.update_playhead(self._cue_seconds)
        self._spectrogram_view.update_cue_marker(self._cue_seconds)

    def _on_reset(self) -> None:
        self._reset_current_track_state(status_key="ui.status.reset")

    def _on_seek(self, seconds: float) -> None:
        # ``seconds`` lives on the ORIGINAL timeline (spectrogram never warps).
        # Convert to playback (processed) timeline before seeking.
        self._cue_seconds = seconds
        self._player.seek(seconds / self._speed_rate())
        self._spectrogram_view.update_cue_marker(seconds)
        self._spectrogram_view.update_playhead(seconds)

    def _on_playback_seek(self, seconds: float) -> None:
        # Temporary seek while playing: move only the live playhead.
        self._player.seek(seconds / self._speed_rate())
        self._spectrogram_view.update_playhead(seconds)

    def _on_selection_changed(self, start: float, end: float) -> None:
        self._apply_loop_region(start, end)
        self._clear_selection_action.setEnabled(True)
        mode_key = (
            "ui.status.selection_mode.loop"
            if self._loop_cb.isChecked()
            else "ui.status.selection_mode.play_once"
        )
        self._set_status(
            "ui.status.selection",
            mode_key=mode_key,
            start=start,
            end=end,
            duration=end - start,
        )

    def _on_selection_cleared(self) -> None:
        self._player.set_loop_enabled(False)
        self._player.set_play_region_enabled(False)
        self._player.set_loop_region(None, None)
        self._clear_selection_action.setEnabled(False)
        self._set_status("ui.status.selection_cleared")

    def _apply_loop_region(self, start: float, end: float) -> None:
        rate = self._speed_rate()
        self._player.set_loop_region(start / rate, end / rate)
        self._player.set_play_region_enabled(False)
        self._player.set_loop_enabled(self._loop_cb.isChecked())

    def _prepare_selection_playback(self) -> None:
        selection = self._spectrogram_view.selection_range()
        if selection is None:
            self._player.set_play_region_enabled(False)
            self._player.set_loop_enabled(False)
            return

        start, end = selection
        self._apply_loop_region(start, end)
        self._player.seek(start / self._speed_rate())
        self._spectrogram_view.update_playhead(start)
        if not self._loop_cb.isChecked():
            self._player.set_play_region_enabled(True)

    def _on_loop_toggled(self, enabled: bool) -> None:
        selection = self._spectrogram_view.selection_range()
        if selection is None:
            self._player.set_loop_enabled(False)
            self._player.set_play_region_enabled(False)
            return

        self._apply_loop_region(*selection)
        if self._player.is_playing:
            self._player.set_play_region_enabled(not enabled)
        mode_key = (
            "ui.status.selection_mode.loop"
            if enabled
            else "ui.status.selection_mode.play_once"
        )
        start, end = selection
        self._set_status(
            "ui.status.selection",
            mode_key=mode_key,
            start=start,
            end=end,
            duration=end - start,
        )

    def _tick(self) -> None:
        position = self._player.position_seconds() * self._speed_rate()
        self._spectrogram_view.update_playhead(position)
        if not self._player.is_playing:
            self._playhead_timer.stop()
            self._player.set_play_region_enabled(False)
            self._spectrogram_view.set_playback_active(False)
            self._play_action.setText(self._tr("ui.action.play"))

    def _refresh_audio_output_device(self) -> None:
        was_playing = self._player.is_playing
        try:
            changed = self._player.refresh_output_device()
        except Exception:
            if was_playing:
                self._playhead_timer.stop()
                self._spectrogram_view.set_playback_active(False)
                self._play_action.setText(self._tr("ui.action.play"))
            return
        if changed and was_playing and self._player.is_playing:
            self._playhead_timer.start()
            self._spectrogram_view.set_playback_active(True)
            self._play_action.setText(self._tr("ui.action.pause"))

    def _on_speed_changed(self, percent: int) -> None:
        # Real time-stretch: schedule a rebuild of the playback buffer.
        self._set_status("ui.status.speed_changed", percent=percent)
        self._audio_transpose_timer.start()

    def _on_audio_transpose_changed(self, _value: float) -> None:
        # Debounce: spinbox emits on every keystroke / arrow press.
        self._audio_transpose_timer.start()

    def _on_language_changed(self, index: int) -> None:
        language = self._language_combo.itemData(index)
        if not isinstance(language, str) or language == self._i18n.language:
            return
        self._i18n.set_language(language)
        self._spectrogram_view.set_i18n(self._i18n)
        self._retranslate_ui()

    def _speed_rate(self) -> float:
        return max(0.05, min(8.0, self._speed_spin.value() / 100.0))

    def _on_temperament_changed(self, *_args) -> None:
        kind = self._temperament_combo.currentData() or "equal"
        is_just = kind == "just"
        self._tonic_label.setEnabled(is_just)
        self._tonic_freq.setEnabled(is_just)
        self._spectrogram_view.set_temperament(
            Temperament(kind=kind, tonic_freq=float(self._tonic_freq.value()))
        )

    def _apply_audio_transpose(self) -> None:
        """Rebuild the playback buffer = pitch-shift ∘ time-stretch.

        Spectrogram is never touched. The original-timeline position of the
        playhead is preserved across the swap.
        """
        if self._current_track is None:
            return
        semitones = float(self._audio_transpose.value())
        rate = self._speed_rate()
        # Original-timeline position of the playhead BEFORE the swap.
        original_position = self._player.position_seconds() * self._previous_rate
        was_playing = self._player.is_playing

        if abs(semitones) < 1e-3 and abs(rate - 1.0) < 1e-3:
            playback_track = self._current_track
        else:
            try:
                QtWidgets.QApplication.setOverrideCursor(
                    QtCore.Qt.CursorShape.WaitCursor
                )
                import librosa
                import numpy as np
                y = self._current_track.samples
                if abs(semitones) >= 1e-3:
                    y = librosa.effects.pitch_shift(
                        y=y,
                        sr=self._current_track.sample_rate,
                        n_steps=semitones,
                    )
                if abs(rate - 1.0) >= 1e-3:
                    y = librosa.effects.time_stretch(y=y, rate=rate)
                y = np.ascontiguousarray(y, dtype=np.float32)
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QApplication.restoreOverrideCursor()
                QtWidgets.QMessageBox.warning(
                    self, self._tr("ui.dialog.audio_processing_error_title"), str(exc)
                )
                return
            else:
                QtWidgets.QApplication.restoreOverrideCursor()
            playback_track = AudioTrack(
                path=self._current_track.path,
                samples=y,
                sample_rate=self._current_track.sample_rate,
            )

        self._player.load(playback_track)
        self._player.seek(original_position / rate)
        # Re-apply the loop region in the new (processed) timeline.
        selection = self._spectrogram_view.selection_range()
        if selection is not None:
            self._apply_loop_region(*selection)
        if was_playing:
            self._player.play()
            self._playhead_timer.start()
        self._previous_rate = rate
        self._set_status(
            "ui.status.playback_settings",
            rate=int(rate * 100),
            semitones=semitones,
        )

    def _on_harmonics_toggled(self, enabled: bool) -> None:
        self._sharpness_slider.setEnabled(enabled)
        self._sharpness_label.setEnabled(enabled)
        if self._current_spec is None:
            return
        if enabled:
            self._recompute_suppressed()
        else:
            self._spectrogram_view.show_spectrogram(
                self._current_spec, preserve_view=True
            )

    def _on_sharpness_changed(self, _value: int) -> None:
        if not self._suppress_harmonics_cb.isChecked():
            return
        self._sharpness_timer.start()

    def _recompute_suppressed(self) -> None:
        if self._current_spec is None:
            return
        # Map slider [0..100] to rel_threshold_db [-50..-5]:
        # higher value → stricter cut → fewer / sharper peaks.
        sharpness = self._sharpness_slider.value()
        threshold_db = -50.0 + (sharpness / 100.0) * 45.0
        try:
            QtWidgets.QApplication.setOverrideCursor(
                QtCore.Qt.CursorShape.WaitCursor
            )
            self._suppressed_spec = suppress_harmonics(
                self._current_spec, rel_threshold_db=threshold_db
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._spectrogram_view.show_spectrogram(
            self._suppressed_spec, preserve_view=True
        )

    def _tr(self, key: str, **params: object) -> str:
        return self._i18n.t(key, **params)

    def _set_status(self, key: str, **params: object) -> None:
        self._status_key = key
        self._status_params = params
        self._update_status_text()

    def _update_status_text(self) -> None:
        self.statusBar().showMessage(
            self._tr(self._status_key, **self._status_format_params(self._status_params))
        )

    def _status_format_params(self, params: dict[str, object]) -> dict[str, object]:
        resolved: dict[str, object] = dict(params)
        mode_key = resolved.pop("mode_key", None)
        if isinstance(mode_key, str):
            resolved["mode"] = self._tr(mode_key)
        resolved.setdefault("seconds_unit", self._tr("ui.unit.seconds_short"))
        resolved.setdefault("hz_unit", self._tr("ui.unit.hz_short"))
        return resolved

    def _retranslate_ui(self) -> None:
        self.setWindowTitle(self._tr("ui.window.title"))
        self._main_toolbar.setWindowTitle(self._tr("ui.toolbar.main"))
        self._tuning_toolbar.setWindowTitle(self._tr("ui.toolbar.tuning"))

        self._open_action.setText(self._tr("ui.action.open"))
        if not self._player.is_playing:
            self._play_action.setText(self._tr("ui.action.play"))
        else:
            self._play_action.setText(self._tr("ui.action.pause"))
        self._stop_action.setText(self._tr("ui.action.stop"))
        self._reset_action.setText(self._tr("ui.action.reset"))
        self._fit_action.setText(self._tr("ui.action.fit"))
        self._clear_selection_action.setText(self._tr("ui.action.clear_selection"))

        self._suppress_harmonics_cb.setText(self._tr("ui.checkbox.suppress_harmonics"))
        self._suppress_harmonics_cb.setToolTip(self._tr("ui.tooltip.suppress_harmonics"))
        self._sharpness_slider.setToolTip(self._tr("ui.tooltip.sharpness"))
        self._speed_spin.setSuffix(self._tr("ui.unit.percent_suffix"))
        self._speed_spin.setToolTip(self._tr("ui.tooltip.speed"))
        self._speed_reset_btn.setToolTip(self._tr("ui.tooltip.speed_reset"))
        self._loop_cb.setText(self._tr("ui.checkbox.loop"))
        self._loop_cb.setToolTip(self._tr("ui.tooltip.loop"))

        self._transpose_indicator.setToolTip(self._tr("ui.tooltip.label_transpose"))
        self._transpose_indicator_reset_btn.setToolTip(
            self._tr("ui.tooltip.label_transpose_reset")
        )
        self._audio_transpose.setToolTip(self._tr("ui.tooltip.audio_transpose"))
        self._audio_transpose_reset_btn.setToolTip(
            self._tr("ui.tooltip.audio_transpose_reset")
        )

        self._temperament_combo.setItemText(0, self._tr("ui.temperament.equal"))
        self._temperament_combo.setItemText(1, self._tr("ui.temperament.just"))
        self._temperament_combo.setToolTip(self._tr("ui.tooltip.temperament"))
        self._tonic_freq.setSuffix(self._tr("ui.unit.hz_suffix"))
        self._tonic_freq.setToolTip(self._tr("ui.tooltip.tonic_freq"))

        self._sharpness_label.setText(self._tr("ui.label.sharpness"))
        self._speed_label.setText(self._tr("ui.label.speed"))
        self._temperament_label.setText(self._tr("ui.label.temperament"))
        self._tonic_label.setText(self._tr("ui.label.tonic_freq"))
        self._label_transpose_label.setText(self._tr("ui.label.label_transpose"))
        self._audio_transpose_label.setText(self._tr("ui.label.audio_transpose"))
        self._language_label.setText(self._tr("ui.label.language"))

        self._language_combo.setItemText(0, self._tr("ui.language.uk"))
        self._language_combo.setItemText(1, self._tr("ui.language.en"))

        self._drop_hint.setText(self._tr("ui.drop_hint"))
        self._spectrogram_view.retranslate_ui()
        self._update_status_text()

    def _reset_current_track_state(self, *, status_key: str | None = None) -> None:
        self._sharpness_timer.stop()
        self._audio_transpose_timer.stop()

        with QtCore.QSignalBlocker(self._suppress_harmonics_cb):
            self._suppress_harmonics_cb.setChecked(False)
        with QtCore.QSignalBlocker(self._sharpness_slider):
            self._sharpness_slider.setValue(self.DEFAULT_SHARPNESS)
        with QtCore.QSignalBlocker(self._speed_spin):
            self._speed_spin.setValue(self.DEFAULT_SPEED_PERCENT)
        with QtCore.QSignalBlocker(self._loop_cb):
            self._loop_cb.setChecked(self.DEFAULT_LOOP_ENABLED)
        with QtCore.QSignalBlocker(self._transpose_indicator):
            self._transpose_indicator.setValue(self.DEFAULT_LABEL_TRANSPOSE)
        with QtCore.QSignalBlocker(self._audio_transpose):
            self._audio_transpose.setValue(self.DEFAULT_AUDIO_TRANSPOSE)
        with QtCore.QSignalBlocker(self._temperament_combo):
            self._temperament_combo.setCurrentIndex(self.DEFAULT_TEMPERAMENT_INDEX)
        with QtCore.QSignalBlocker(self._tonic_freq):
            self._tonic_freq.setValue(self.DEFAULT_TONIC_FREQ)

        self._sharpness_slider.setEnabled(False)
        self._sharpness_label.setEnabled(False)

        self._spectrogram_view.set_label_transpose(self.DEFAULT_LABEL_TRANSPOSE)
        self._on_temperament_changed()

        self._cue_seconds = 0.0
        self._player.stop()
        self._player.set_loop_enabled(False)
        self._player.set_play_region_enabled(False)
        self._player.set_loop_region(None, None)
        self._playhead_timer.stop()
        self._spectrogram_view.set_playback_active(False)
        self._play_action.setText(self._tr("ui.action.play"))

        if self._current_track is not None:
            self._player.load(self._current_track)
        self._previous_rate = 1.0
        self._suppressed_spec = None

        if self._current_spec is not None:
            self._spectrogram_view.show_spectrogram(self._current_spec)
            self._spectrogram_view.fit_view()
            self._spectrogram_view.clear_selection()
            self._spectrogram_view.update_playhead(0.0)
            self._spectrogram_view.update_cue_marker(0.0)

        self._clear_selection_action.setEnabled(False)
        if status_key is not None:
            self._set_status(status_key)

    # ----------------------------------------------------------------- loader
    def _load_path(self, path: Path) -> None:
        self._set_status("ui.status.loading", file_name=path.name)
        QtWidgets.QApplication.processEvents()

        try:
            track = load_track(path)
            spectrogram = compute_spectrogram(track)
        except Exception as exc:  # noqa: BLE001 — surface any backend failure to the user
            QtWidgets.QMessageBox.critical(
                self,
                self._tr("ui.dialog.open_failed_title"),
                f"{path.name}\n\n{exc}",
            )
            self._set_status("ui.status.load_error")
            return

        self._install_track(track, spectrogram)

    def _install_track(self, track: AudioTrack, spectrogram: Spectrogram) -> None:
        self._current_track = track
        self._current_spec = spectrogram
        self._suppressed_spec = None

        self._reset_current_track_state()

        self._stack.setCurrentWidget(self._spectrogram_view)
        self._play_action.setEnabled(True)
        self._stop_action.setEnabled(True)
        self._reset_action.setEnabled(True)
        self._fit_action.setEnabled(True)
        self._clear_selection_action.setEnabled(False)
        self._play_action.setText(self._tr("ui.action.play"))
        self._set_status(
            "ui.status.track_loaded",
            file_name=track.path.name,
            duration=track.duration_seconds,
            sample_rate=track.sample_rate,
        )
