# coding=utf-8
import sys
from PyQt5.QtWidgets import QApplication

from main_windows.FER_Window import FERWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName("My FER Program")

    window = FERWindow()
    app.exec_()