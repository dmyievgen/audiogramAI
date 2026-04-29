"""Application entry point."""
from __future__ import annotations

import sys

from PyQt6 import QtWidgets

from .ui.main_window import MainWindow


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Audiogram")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
