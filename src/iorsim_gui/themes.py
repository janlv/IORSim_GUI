
from PySide6.QtWidgets import QStyleFactory
from PySide6.QtGui import QPalette, QColor, QFont
from PySide6.QtCore import Qt

def set_light_theme(app):
    app.setStyle(QStyleFactory.create("Fusion"))
    palette = QPalette()

    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, Qt.black)
    palette.setColor(QPalette.Base, Qt.white)
    palette.setColor(QPalette.AlternateBase, QColor(225, 225, 225))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.black)
    palette.setColor(QPalette.Text, Qt.black)
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, Qt.black)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(0, 120, 215))
    palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
    palette.setColor(QPalette.HighlightedText, Qt.white)

    # Set global font
    font = QFont("Sans Serif", 11)
    app.setFont(font)

    app.setPalette(palette)


