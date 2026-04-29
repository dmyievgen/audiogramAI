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
from .spectrogram_view import SpectrogramView


class MainWindow(QtWidgets.QMainWindow):
    PLAYHEAD_INTERVAL_MS = 30
    DEFAULT_SHARPNESS = 50
    DEFAULT_SPEED_PERCENT = 100
    DEFAULT_LABEL_TRANSPOSE = 0.0
    DEFAULT_AUDIO_TRANSPOSE = 0.0
    DEFAULT_TEMPERAMENT_INDEX = 0
    DEFAULT_TONIC_FREQ = 440.0

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Audiogram")
        self.resize(1180, 640)
        self.setAcceptDrops(True)

        self._player = AudioPlayer()

        self._open_action = QtGui.QAction("Open…", self)
        self._open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self._open_action.triggered.connect(self._on_open_dialog)

        self._play_action = QtGui.QAction("Play", self)
        self._play_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Space))
        self._play_action.setEnabled(False)
        self._play_action.triggered.connect(self._on_toggle_playback)

        self._stop_action = QtGui.QAction("Stop", self)
        self._stop_action.setEnabled(False)
        self._stop_action.triggered.connect(self._on_stop)

        self._reset_action = QtGui.QAction("Reset", self)
        self._reset_action.setEnabled(False)
        self._reset_action.triggered.connect(self._on_reset)

        self._fit_action = QtGui.QAction("Fit", self)
        self._fit_action.setShortcut(QtGui.QKeySequence("Ctrl+0"))
        self._fit_action.setEnabled(False)
        self._fit_action.triggered.connect(lambda: self._spectrogram_view.fit_view())

        self._clear_selection_action = QtGui.QAction("Очистити виділення", self)
        self._clear_selection_action.setEnabled(False)
        self._clear_selection_action.triggered.connect(
            lambda: self._spectrogram_view.clear_selection()
        )

        self._suppress_harmonics_cb = QtWidgets.QCheckBox("Без гармонік")
        self._suppress_harmonics_cb.setToolTip(
            "Виділити лише основні тони (прибрати обертони)"
        )
        self._suppress_harmonics_cb.toggled.connect(self._on_harmonics_toggled)

        self._sharpness_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._sharpness_slider.setRange(0, 100)
        self._sharpness_slider.setValue(50)
        self._sharpness_slider.setFixedWidth(120)
        self._sharpness_slider.setEnabled(False)
        self._sharpness_slider.setToolTip(
            "Гострота виділення основних тонів: ліворуч — більше точок, "
            "праворуч — тільки найяскравіші піки"
        )
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
        self._speed_spin.setSuffix(" %")
        self._speed_spin.setToolTip(
            "Швидкість прослуховування (varispeed: міняє і темп, і висоту)."
        )
        self._speed_spin.valueChanged.connect(self._on_speed_changed)

        self._speed_reset_btn = QtWidgets.QToolButton()
        self._speed_reset_btn.setText("↺")
        self._speed_reset_btn.setToolTip("Повернути швидкість до 100%")
        self._speed_reset_btn.clicked.connect(lambda: self._speed_spin.setValue(100))

        self._transpose_indicator = QtWidgets.QDoubleSpinBox()
        self._transpose_indicator.setDecimals(2)
        self._transpose_indicator.setRange(-12.0, 12.0)
        self._transpose_indicator.setSingleStep(1.0)
        self._transpose_indicator.setValue(0.0)
        self._transpose_indicator.setToolTip(
            "Транспонування підпису ноти під курсором (візуально — спектр не змінюється)"
        )
        self._transpose_indicator.valueChanged.connect(
            lambda value: self._spectrogram_view.set_label_transpose(value)
        )
        self._transpose_indicator_reset_btn = QtWidgets.QToolButton()
        self._transpose_indicator_reset_btn.setText("↺")
        self._transpose_indicator_reset_btn.setToolTip("Повернути транспонацію нот до 0")
        self._transpose_indicator_reset_btn.clicked.connect(
            lambda: self._transpose_indicator.setValue(0.0)
        )

        self._audio_transpose = QtWidgets.QDoubleSpinBox()
        self._audio_transpose.setDecimals(2)
        self._audio_transpose.setRange(-12.0, 12.0)
        self._audio_transpose.setSingleStep(1.0)
        self._audio_transpose.setValue(0.0)
        self._audio_transpose.setToolTip(
            "Пітч-шифт лише для відтворення. Спектрограма і ноти залишаються без змін."
        )
        self._audio_transpose.valueChanged.connect(self._on_audio_transpose_changed)
        self._audio_transpose_timer = QtCore.QTimer(self)
        self._audio_transpose_timer.setSingleShot(True)
        self._audio_transpose_timer.setInterval(250)
        self._audio_transpose_timer.timeout.connect(self._apply_audio_transpose)
        self._audio_transpose_reset_btn = QtWidgets.QToolButton()
        self._audio_transpose_reset_btn.setText("↺")
        self._audio_transpose_reset_btn.setToolTip("Повернути транспонацію звуку до 0")
        self._audio_transpose_reset_btn.clicked.connect(
            lambda: self._audio_transpose.setValue(0.0)
        )

        self._temperament_combo = QtWidgets.QComboBox()
        self._temperament_combo.addItem("Рівнотемперована", "equal")
        self._temperament_combo.addItem("Натуральна", "just")
        self._temperament_combo.setToolTip(
            "Темперація впливає лише на позиції та підписи нот (спектр без змін)."
        )
        self._temperament_combo.currentIndexChanged.connect(self._on_temperament_changed)

        self._tonic_freq = QtWidgets.QDoubleSpinBox()
        self._tonic_freq.setDecimals(1)
        self._tonic_freq.setRange(20.0, 8000.0)
        self._tonic_freq.setSingleStep(0.5)
        self._tonic_freq.setSuffix(" Гц")
        self._tonic_freq.setValue(440.0)
        self._tonic_freq.setToolTip(
            "Частота тоніки натуральної темперації (напр. 440 Гц = A4, 435 Гц = «old pitch»)."
        )
        self._tonic_freq.valueChanged.connect(self._on_temperament_changed)
        self._tonic_label = QtWidgets.QLabel(" тоніка (Ч): ")

        toolbar = self.addToolBar("Main")
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
        self._sharpness_label = QtWidgets.QLabel(" гострота: ")
        self._sharpness_label.setEnabled(False)
        toolbar.addWidget(self._sharpness_label)
        toolbar.addWidget(self._sharpness_slider)
        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel(" Швидкість: "))
        toolbar.addWidget(self._speed_spin)
        toolbar.addWidget(self._speed_reset_btn)

        # Force a second row so transposition + temperament always fit.
        self.addToolBarBreak()
        toolbar2 = self.addToolBar("Tuning")
        toolbar2.setMovable(False)
        toolbar2.setFloatable(False)
        toolbar2.layout().setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        toolbar2.addWidget(QtWidgets.QLabel(" Темперація: "))
        toolbar2.addWidget(self._temperament_combo)
        toolbar2.addWidget(self._tonic_label)
        toolbar2.addWidget(self._tonic_freq)
        toolbar2.addSeparator()
        toolbar2.addWidget(QtWidgets.QLabel(" Транспонація нот (півтони): "))
        toolbar2.addWidget(self._transpose_indicator)
        toolbar2.addWidget(self._transpose_indicator_reset_btn)
        toolbar2.addSeparator()
        toolbar2.addWidget(QtWidgets.QLabel(" Транспонація звуку (півтони): "))
        toolbar2.addWidget(self._audio_transpose)
        toolbar2.addWidget(self._audio_transpose_reset_btn)

        self._spectrogram_view = SpectrogramView()
        self._spectrogram_view.seek_requested.connect(self._on_seek)
        self._spectrogram_view.playback_seek_requested.connect(
            self._on_playback_seek
        )
        self._spectrogram_view.selection_changed.connect(self._on_selection_changed)
        self._spectrogram_view.selection_cleared.connect(self._on_selection_cleared)
        # Apply initial temperament + tonic-combo visibility.
        self._on_temperament_changed()

        self._drop_hint = QtWidgets.QLabel(
            "Перетягни сюди аудіофайл або натисни Open…"
        )
        self._drop_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._drop_hint.setStyleSheet(
            "color: #b4c7a8; font-size: 16px; padding: 24px;"
        )

        self._stack = QtWidgets.QStackedWidget()
        self._stack.addWidget(self._drop_hint)
        self._stack.addWidget(self._spectrogram_view)
        self.setCentralWidget(self._stack)

        self.statusBar().showMessage("Готово")

        self._playhead_timer = QtCore.QTimer(self)
        self._playhead_timer.setInterval(self.PLAYHEAD_INTERVAL_MS)
        self._playhead_timer.timeout.connect(self._tick)

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
        filters = "Audio (" + " ".join(f"*{s}" for s in sorted(SUPPORTED_SUFFIXES)) + ")"
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Виберіть аудіофайл", "", filters
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
            self._play_action.setText("Play")
        else:
            self._player.play()
            if self._player.is_playing:
                self._playhead_timer.start()
                self._spectrogram_view.set_playback_active(True)
                self._play_action.setText("Pause")

    def _on_stop(self) -> None:
        self._player.stop()
        self._playhead_timer.stop()
        self._player.seek(self._cue_seconds / self._speed_rate())
        self._play_action.setText("Play")
        self._spectrogram_view.set_playback_active(False)
        self._spectrogram_view.update_playhead(self._cue_seconds)
        self._spectrogram_view.update_cue_marker(self._cue_seconds)

    def _on_reset(self) -> None:
        self._reset_current_track_state(status_message="Скинуто до початкового стану")

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
        self.statusBar().showMessage(
            f"Виділення (loop): {start:.2f} – {end:.2f} с ({end - start:.2f} с)"
        )

    def _on_selection_cleared(self) -> None:
        self._player.set_loop_enabled(False)
        self._player.set_loop_region(None, None)
        self._clear_selection_action.setEnabled(False)
        self.statusBar().showMessage("Виділення скинуто")

    def _apply_loop_region(self, start: float, end: float) -> None:
        rate = self._speed_rate()
        self._player.set_loop_region(start / rate, end / rate)
        self._player.set_loop_enabled(True)

    def _tick(self) -> None:
        position = self._player.position_seconds() * self._speed_rate()
        self._spectrogram_view.update_playhead(position)
        if not self._player.is_playing:
            self._playhead_timer.stop()
            self._spectrogram_view.set_playback_active(False)
            self._play_action.setText("Play")

    def _on_speed_changed(self, percent: int) -> None:
        # Real time-stretch: schedule a rebuild of the playback buffer.
        self.statusBar().showMessage(f"Швидкість прослуховування: {percent}% (без зміни тону)")
        self._audio_transpose_timer.start()

    def _on_audio_transpose_changed(self, _value: float) -> None:
        # Debounce: spinbox emits on every keystroke / arrow press.
        self._audio_transpose_timer.start()

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
                    self, "Помилка обробки аудіо", str(exc)
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
        self.statusBar().showMessage(
            f"Відтворення: {int(rate * 100)}% · транспонація звуку {semitones:+.2f} півтона"
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

    def _reset_current_track_state(self, *, status_message: str | None = None) -> None:
        self._sharpness_timer.stop()
        self._audio_transpose_timer.stop()

        with QtCore.QSignalBlocker(self._suppress_harmonics_cb):
            self._suppress_harmonics_cb.setChecked(False)
        with QtCore.QSignalBlocker(self._sharpness_slider):
            self._sharpness_slider.setValue(self.DEFAULT_SHARPNESS)
        with QtCore.QSignalBlocker(self._speed_spin):
            self._speed_spin.setValue(self.DEFAULT_SPEED_PERCENT)
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
        self._playhead_timer.stop()
        self._spectrogram_view.set_playback_active(False)
        self._play_action.setText("Play")

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
        if status_message is not None:
            self.statusBar().showMessage(status_message)

    # ----------------------------------------------------------------- loader
    def _load_path(self, path: Path) -> None:
        self.statusBar().showMessage(f"Завантажую {path.name}…")
        QtWidgets.QApplication.processEvents()

        try:
            track = load_track(path)
            spectrogram = compute_spectrogram(track)
        except Exception as exc:  # noqa: BLE001 — surface any backend failure to the user
            QtWidgets.QMessageBox.critical(
                self, "Не вдалося відкрити файл", f"{path.name}\n\n{exc}"
            )
            self.statusBar().showMessage("Помилка завантаження")
            return

        self._install_track(
            track,
            spectrogram,
            label=f"{path.name} · {track.duration_seconds:.1f} с · {track.sample_rate} Гц",
        )

    def _install_track(self, track: AudioTrack, spectrogram: Spectrogram, *, label: str) -> None:
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
        self._play_action.setText("Play")
        self.statusBar().showMessage(label)
