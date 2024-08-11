import sys
from PyQt5.QtWidgets import (QApplication)
from PyQt5.QtGui import QColor, QPalette

from spinepipe.Scripts.SpinePipe_GUI_v2_dual import Splash, MainWindow, __version__


def main():
    app = QApplication([])

    #app.setStyleSheet(qss)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(230, 230, 230))
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(204, 204, 204))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.Button, QColor(204, 204, 204))
    palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(0, 102, 153))
    palette.setColor(QPalette.Highlight, QColor(0, 102, 153))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

    app.setPalette(palette)


    splash = Splash("SpinePipe loading...", 2000)
    splash.move(600, 600)

    splash.show()

    # Ensures that the application is fully up and running before closing the splash screen
    app.processEvents()

    window = MainWindow()
    window.setWindowTitle(f' SpinePipe - Version: {__version__}')
    window.setGeometry(100, 100, 1200, 1200)  
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()