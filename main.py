"""SHINOBI VISION X entry point."""
import sys


def main():
    if "--diagnose" in sys.argv:
        from core.diagnostics import print_report
        sys.exit(0 if print_report() else 1)
    try:
        from PySide6.QtWidgets import QApplication
        from app.main_window import MainWindow
        from app.theme import QSS
        from core.config import Settings
    except ImportError as e:
        print(f"Missing dependency: {e}\nRun:  pip install -r requirements.txt")
        sys.exit(1)
    qt = QApplication(sys.argv)
    qt.setStyleSheet(QSS)
    win = MainWindow(Settings())
    win.show()
    sys.exit(qt.exec())


if __name__ == "__main__":
    main()
