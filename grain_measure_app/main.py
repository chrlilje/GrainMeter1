from PySide6.QtWidgets import QApplication
import sys

try:
    from .app_window import AppWindow
except ImportError:
    from app_window import AppWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
