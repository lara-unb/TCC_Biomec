from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTime, QTimer, QSize
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget, QTableWidgetItem, QVBoxLayout, QDialog
from interface import Ui_MainWindow
from file_opening import Ui_FileWindow
import sys
import cv2
# sys.path.append("../src")
# from support import readFileDialog

class MainWindow_EXEC():
    def __init__(self):
        app = QtWidgets.QApplication(sys.argv)
        
        self.setup_MainWindow()
        self.setup_FileWindow()

        self.MainWindow.show()
        sys.exit(app.exec_())
    
    def setup_MainWindow(self):
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)

    def setup_FileWindow(self):
        self.FileWindow = QtWidgets.QMainWindow()
        self.ui_fw = Ui_FileWindow()
        self.ui_fw.setupUi(self.FileWindow)
        self.opened_files = []
        self.ui.actionOpen.triggered.connect(self.file_opening_dialog)
        self.ui_fw.browse_file.clicked.connect(self.browse_file)
        self.ui_fw.add_file.clicked.connect(self.add_file)
        self.ui_fw.rm_file.clicked.connect(self.rm_file)
        self.ui_fw.files_list_wg.currentItemChanged.connect(self.files_list_cur_changed)

    def file_opening_dialog(self):
        self.FileWindow.show()
    
    def browse_file(self):
        print("Opening File")
        file_path = self.readFileDialog("Open data file")
        self.ui_fw.file_path.setText(file_path)

    def add_file(self):
        text = self.ui_fw.file_path.text()
        if text.split(".")[-1] == ".data":
            parse_data_file

        if text not in self.opened_files:
            self.opened_files.append(text)
        self.ui_fw.files_list_wg.clear()
        self.ui_fw.files_list_wg.addItems(self.opened_files)
    
    def rm_file(self):
        text = self.ui_fw.file_path.text()
        if text in self.opened_files:
            self.opened_files.remove(text)
        self.ui_fw.files_list_wg.clear()
        self.ui_fw.files_list_wg.addItems(self.opened_files)

    def files_list_cur_changed(self, current):
        try:
            item = current.text()
            self.ui_fw.file_path.setText(item)
        except:
            pass

    def readFileDialog(self, title="Open File", file_type="All Files"):
        if file_type == "All Files":
            type_filter = "All Files (*)"
        else:
            type_filter = file_type + " (*." + file_type + ")"
        self.options = QtWidgets.QFileDialog.Options()
        self.options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(None,title,
                                                            "", type_filter, 
                                                            options=self.options)
        return file_name

if __name__ == "__main__":
    MainWindow_EXEC()