from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTime, QTimer, QSize
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget, QTableWidgetItem, QVBoxLayout, QDialog, QMessageBox
from interface import Ui_MainWindow
from file_opening import Ui_FileWindow
from plot_settings import Ui_PlotWindow
from data_opening import Ui_DataWindow
import importlib
import sys
import cv2
import numpy as np
from copy import deepcopy
import pickle
sys.path.append("../src")
from support import parse_data_file, parse_pickle_file, function_from_file, save_to_file

class myPlot():
    def __init__(self, plot_id, title, y_label, x_label, segment=False):
        self.plot_id = plot_id
        self.title = title
        self.y_label = y_label
        self.x_label = x_label
        self.data_plot = {}
        self.data = {}
        self.segment = segment
    def activeData(self):
        return self.data.keys()
    def removeData(self, data_id, element):
        self.data.remove(data_id)

class MainWindow_EXEC():
    def __init__(self):
        app = QtWidgets.QApplication(sys.argv)

        self.setup_MainWindow()
        self.setup_FileWindow()
        self.setup_PlotWindow()
        self.setup_DataWindow()

        self.video_created = False
        self.time_vid = 0
        self.plot_interval = 20
        self.save_file_path = ""

        self.ui.actionOpen.triggered.connect(self.file_open_dialog)
        self.ui.actionSave.triggered.connect(self.save_dialog)
        self.ui.actionSave_as.triggered.connect(self.save_as_dialog)

        self.MainWindow.show()
        sys.exit(app.exec_())
# Windows Setup ---------------------------------------------------------------------------------------------------------------------
    def setup_MainWindow(self):
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)
    def setup_FileWindow(self):
        self.FileWindow = QtWidgets.QMainWindow()
        self.ui_fw = Ui_FileWindow()
        self.ui_fw.setupUi(self.FileWindow)
        self.selected_data_dic = {}
        self.opened_files = {}
        self.ui.actionAddFiles.triggered.connect(self.file_add_files_dialog)
        self.ui_fw.browse_file.clicked.connect(self.browse_file)
        self.ui_fw.add_file.clicked.connect(self.add_file)
        self.ui_fw.rm_file.clicked.connect(self.rm_file)
        self.ui_fw.files_list_wg.currentItemChanged.connect(self.files_list_cur_changed)
        self.ui_fw.available_data.currentItemChanged.connect(self.available_data_cur_changed)
        self.ui_fw.selected_data.currentItemChanged.connect(self.selected_data_cur_changed)
        self.ui_fw.select_data.clicked.connect(self.select_data)
        self.ui_fw.rm_data.clicked.connect(self.rm_data)
    def setup_PlotWindow(self):
        self.PlotWindow = QtWidgets.QMainWindow()
        self.ui_pw = Ui_PlotWindow()
        self.ui_pw.setupUi(self.PlotWindow)

        self.size_of_graph = 1000
        self.t0 = np.zeros(self.size_of_graph)

        self.max_plots = 4
        self.window_plots = {"Upper": self.ui.window_plot_upper, "Lower": self.ui.window_plot_lower}
        self.plots_vec = {"Upper": {}, "Lower": {}, "Inactive": {}}

        self.ui.actionPlots.triggered.connect(self.plot_settings_dialog)
        self.ui_pw.ca_plot_button.clicked.connect(self.on_ca_plot_button_clicked)
        self.ui_pw.rm_plot_button.clicked.connect(self.on_rm_plot_button_clicked)
        self.ui_pw.add_data_button.clicked.connect(self.on_add_data_button_clicked)
        self.ui_pw.rm_data_button.clicked.connect(self.on_rm_data_button_clicked)
        self.ui_pw.inactive_plots.itemDoubleClicked.connect(self.inactive_plots_cur_changed)
        self.ui_pw.upper_plots.itemDoubleClicked.connect(self.upper_plots_cur_changed)
        self.ui_pw.lower_plots.itemDoubleClicked.connect(self.lower_plots_cur_changed)
        self.ui_pw.inactive_plots.itemChanged.connect(self.inactive_plots_changed)
        self.ui_pw.data_list.itemChanged.connect(self.data_list_cur_changed)
        self.ui_pw.upper_plots.itemChanged.connect(self.upper_plots_changed)
        self.ui_pw.lower_plots.itemChanged.connect(self.lower_plots_changed)
    def setup_DataWindow(self):
        self.DataWindow = QtWidgets.QMainWindow()
        self.ui_dw = Ui_DataWindow()
        self.ui_dw.setupUi(self.DataWindow)

        self.opened_functions_path = {}
        self.opened_functions = {}

        self.ui_dw.browse_function.clicked.connect(self.browse_functions)
        self.ui_dw.add_function.clicked.connect(self.add_function)
        self.ui_dw.rm_function.clicked.connect(self.rm_function)
        self.ui_dw.add_data.clicked.connect(self.add_data)
        self.ui_dw.opened_functions.currentItemChanged.connect(self.opened_functions_cur_changed)
        self.ui_dw.selected_data.currentItemChanged.connect(self.selected_data_cur_changed_dw)
# File Functions --------------------------------------------------------------------------------------------------------------------
    def file_add_files_dialog(self):
        self.FileWindow.show()
    def browse_file(self):
        print("Opening File")
        file_path = self.readFileDialog("Open data file")
        self.ui_fw.file_path.setText(file_path)
    def add_file(self):
        text = self.ui_fw.file_path.text()
        try:
            print("Opening: {}".format(text))
            if text.split(".")[-1] == "data":
                data, var_names = parse_data_file(text)
            elif text.split(".")[-1] == "out":
                data, var_names = parse_pickle_file(text)
            else:
                QMessageBox.about(self.FileWindow, "Warning", "File not recognized")
                print("File not recognized")
                return
        except Exception as e:
            QMessageBox.about(self.FileWindow, "Warning", "File corrupted")
            print("File corrupted")
            print(e)
            return

        if text not in self.opened_files:
            self.opened_files[text] = [data, var_names]
        self.ui_fw.files_list_wg.clear()
        self.ui_fw.files_list_wg.addItems([*self.opened_files])
    def rm_file(self):
        text = self.ui_fw.file_path.text()
        if text in self.opened_files:
            self.opened_files.pop(text)
        self.ui_fw.files_list_wg.clear()
        self.ui_fw.files_list_wg.addItems([*self.opened_files])
    def files_list_cur_changed(self, current):
        if current != None:
            text = current.text()
            self.ui_fw.file_path.setText(text)
            print("Opening: {}".format(text))
            if text.split(".")[-1] == "data":
                self.data, self.var_names = parse_data_file(text)
            elif text.split(".")[-1] == "out":
                self.data, self.var_names = parse_pickle_file(text)
            else:
                print("Odd Data: ", text)
                return
            self.ui_fw.available_data.clear()
            self.ui_fw.available_data.addItems(self.var_names)
            self.ui_fw.available_data.addItem("All")
    def available_data_cur_changed(self, current):
        if current != None:
            text = current.text()
            self.ui_fw.data_avail.setText(text)
    def selected_data_cur_changed(self, current):
        if current != None:
            text = current.text()
            self.ui_fw.select_data_id.setText(text)
    def select_data(self):
        data_id = self.ui_fw.data_avail_id.text()
        data_type = self.ui_fw.data_avail.text()
        if data_type == "All":
            for var in self.var_names:
                self.selected_data_dic[data_id + "_" + str(var)] = self.data[var]
                self.ui_fw.selected_data.addItem(data_id + "_" + str(var))
        else:
            self.selected_data_dic[data_id] = self.data[data_type]
            self.ui_fw.selected_data.addItem(data_id)
    def rm_data(self):
        data_id = self.ui_fw.select_data_id.text()
        self.selected_data_dic.pop(data_id)
        self.ui_fw.selected_data.clear()
        self.ui_fw.selected_data.addItems([*self.selected_data_dic])
# Plot Functions --------------------------------------------------------------------------------------------------------------------
    def plot_settings_dialog(self):
        self.PlotWindow.show()
    def on_ca_plot_button_clicked(self):
        plot_id = self.ui_pw.plot_id.text()
        pos = self.find_pos_from_id(plot_id)
        
        if (pos == "Upper") or (pos == "Lower"):
            print("Altering graph")
            self.window_plots[pos].removeItem(self.plots_vec[pos][plot_id].plot)
        elif pos == "Inactive":
            print("Altering graph")
        else:
            pos = "Inactive"
            print("Creating graph")

        self.plots_vec[pos][plot_id] = myPlot(plot_id = plot_id,
                                            title = self.ui_pw.title_plot.text(),
                                            y_label = self.ui_pw.y_label.text(),
                                            x_label = self.ui_pw.x_label.text())
        if (pos == "Upper") or (pos == "Lower"):
            self.plots_vec[pos][plot_id].plot = self.window_plots[pos].addPlot(title=self.plots_vec[pos][plot_id].title,
                                                                labels = {'left':self.plots_vec[pos][plot_id].y_label, 
                                                                        'bottom':self.plots_vec[pos][plot_id].x_label})
            self.plots_vec[pos][plot_id].plot.addLegend()
        self.update_plot_list()
    def on_rm_plot_button_clicked(self):
        plot_id = self.ui_pw.plot_id.text()
        pos = self.find_pos_from_id(plot_id)
        print(pos)
        try:
            if (pos == "Upper") or (pos == "Lower"):
                self.window_plots[pos].removeItem(self.plots_vec[pos][plot_id].plot)
                self.plots_vec[pos].pop(plot_id)
            elif pos == "Inactive":
                self.plots_vec[pos].pop(plot_id)
            else:
                print("Plot ID invalid")
        except:
            print("Plot ID invalid")
        self.update_plot_list()
    def on_add_data_button_clicked(self):
        self.ui_dw.plot_id.setText(self.ui_pw.plot_id.text())
        self.ui_dw.selected_data.addItems([*self.selected_data_dic])
        self.DataWindow.show()
    def on_rm_data_button_clicked(self):
        plot_id = self.ui_pw.plot_id.text()
        pos = self.find_pos_from_id(plot_id)
        if (pos == "Upper") or (pos == "Lower"):
            print("On rm data")
            print("Pos: {}, Plot ID: {}, Cur Data ID: {}".format(pos, plot_id, self.ui_pw.data_list.currentItem().text()))
            self.plots_vec[pos][plot_id].plot.removeItem(self.plots_vec[pos][plot_id].data_plot[self.ui_pw.data_list.currentItem().text()])
            del self.plots_vec[pos][plot_id].data_plot[self.ui_pw.data_list.currentItem().text()]
        del self.plots_vec[pos][plot_id].data[self.ui_pw.data_list.currentItem().text()]
        self.update_data_list(plot_id)
    def inactive_plots_cur_changed(self, current):
        if current is not None:
            if current.text() != '':
                self.update_plot_settings(current.text(), "Inactive")
                self.update_data_list(current.text())
    def upper_plots_cur_changed(self, current):
        if current is not None:
            if current.text() != '':
                self.update_plot_settings(current.text(), "Upper")
                self.update_data_list(current.text())
    def lower_plots_cur_changed(self, current):
        if current is not None:
            if current.text() != '':
                self.update_plot_settings(current.text(), "Lower")
                self.update_data_list(current.text())
    def inactive_plots_changed(self, item):
        if item.text() != '':
            print("Inactive: {}".format(item.text()))
            self.ui_pw.inactive_plots.itemChanged.disconnect(self.inactive_plots_changed)
            print("Rearrange")
            self.rearrange_plot_vec(item.text(), "Inactive")
            self.ui_pw.inactive_plots.itemChanged.connect(self.inactive_plots_changed)   
    def data_list_cur_changed(self, current):
        if current != None:
            text = current.text()
            print("Data list cur changed: ", text)
    def upper_plots_changed(self, item):
        if item.text() != '':
            print("Upper: {}".format(item.text()))
            self.ui_pw.upper_plots.itemChanged.disconnect(self.upper_plots_changed)
            print("Rearrange")
            self.rearrange_plot_vec(item.text(), "Upper")
            self.ui_pw.upper_plots.itemChanged.connect(self.upper_plots_changed)
    def lower_plots_changed(self, item):
        if item.text() != '':
            print("Lower: {}".format(item.text()))
            self.ui_pw.lower_plots.itemChanged.disconnect(self.lower_plots_changed)
            print("Rearrange")
            self.rearrange_plot_vec(item.text(), "Lower")
            self.ui_pw.lower_plots.itemChanged.connect(self.lower_plots_changed) 

    def update_plot_list(self):
        self.ui_pw.inactive_plots.clear()
        self.ui_pw.inactive_plots.addItems(self.plots_vec["Inactive"].keys())
        self.ui_pw.upper_plots.clear()
        self.ui_pw.upper_plots.addItems(self.plots_vec["Upper"].keys())
        self.ui_pw.lower_plots.clear()
        self.ui_pw.lower_plots.addItems(self.plots_vec["Lower"].keys())
    def rearrange_plot_vec(self, plot_id, pos):
        print("item: {}, pos:{}".format(plot_id, pos))
        if plot_id in self.plots_vec["Inactive"].keys():
            prev_pos = "Inactive"
        elif plot_id in self.plots_vec["Upper"].keys():
            prev_pos = "Upper"
        elif plot_id in self.plots_vec["Lower"].keys():
            prev_pos = "Lower"
        else:
            print("Not found: {}".format(plot_id))
            return
        print("Prev pos:", prev_pos)
        self.plots_vec[pos][plot_id] = self.plots_vec[prev_pos][plot_id]
        if (prev_pos == "Upper") or (prev_pos == "Lower"):
            print("Moving graph")
            self.window_plots[prev_pos].removeItem(self.plots_vec[prev_pos][plot_id].plot)
            self.plots_vec[prev_pos][plot_id].data_plot = {}
        self.plots_vec[prev_pos].pop(plot_id)
        if (pos == "Upper") or (pos == "Lower"):
            self.plots_vec[pos][plot_id].plot = self.window_plots[pos].addPlot(title=self.plots_vec[pos][plot_id].title,
                                                                labels = {'left':self.plots_vec[pos][plot_id].y_label, 
                                                                        'bottom':self.plots_vec[pos][plot_id].x_label})
            self.plots_vec[pos][plot_id].plot.addLegend()
            self.set_data_plot(plot_id)
    def update_plot_settings(self, current, pos):
        self.ui_pw.plot_id.setText(self.plots_vec[pos][current].plot_id)
        self.ui_pw.title_plot.setText(self.plots_vec[pos][current].title)
        self.ui_pw.y_label.setText(self.plots_vec[pos][current].y_label)
        self.ui_pw.x_label.setText(self.plots_vec[pos][current].x_label)
# Data Functions --------------------------------------------------------------------------------------------------------------------
    def browse_functions(self):
        print("Opening Function")
        file_path = self.readFileDialog("Open function file", "py")
        self.ui_dw.function_file_path.setText(file_path)
    def add_function(self):
        function_path = self.ui_dw.function_file_path.text()
        function_id = self.ui_dw.function_id.text()
        print("Opening: {}".format(function_path))
        if function_path not in self.opened_functions_path.values():
            self.opened_functions_path[function_id] = function_path
        self.ui_dw.opened_functions.clear()
        self.ui_dw.opened_functions.addItems([*self.opened_functions_path])
    def rm_function(self):
        function_id = self.ui_dw.function_id.text()
        if function_id in self.opened_functions_path:
            self.opened_functions_path.pop(function_id)
        self.ui_dw.opened_functions.clear()
        self.ui_dw.opened_functions.addItems([*self.opened_functions_path])
    def add_data(self):
        plot_id = self.ui_dw.plot_id.text()
        pos = self.find_pos_from_id(plot_id)
        data_id = self.ui_dw.data_id.text()

        data_vec, data_time = self.get_data_from_data_line(self.ui_dw.data_line.text())
        R, G, B = self.ui_dw.data_R_n.value(), self.ui_dw.data_G_n.value(), self.ui_dw.data_B_n.value()
        label = self.ui_dw.data_label.text()
        self.plots_vec[pos][plot_id].data[data_id] = {"data_vec": data_vec, "data_time": data_time,
                                                "color": [R, G, B], "label": label}        
        self.add_data_to_plot(data_id, plot_id)
        self.update_data_list(plot_id)
    def opened_functions_cur_changed(self, current):
        if current != None:
            function_id = current.text()
            function_path = self.opened_functions_path[function_id]
            print("Funtion ID: ", function_id)
            print("Function path: ", function_path)
            self.ui_dw.function_id.setText(function_id)
            self.ui_dw.function_file_path.setText(function_path)
    def selected_data_cur_changed_dw(self, current):
        text = current.text()
        data = self.selected_data_dic[text]
        self.set_data_info(data)

    def set_data_plot(self, plot_id):
        pos = self.find_pos_from_id(plot_id)
        for data_id in self.plots_vec[pos][plot_id].data.keys():
            self.add_data_to_plot(data_id, plot_id)
    def add_data_to_plot(self, data_id, plot_id):
        pos = self.find_pos_from_id(plot_id)
        data_time = self.plots_vec[pos][plot_id].data[data_id]["data_time"]
        data_vec = self.plots_vec[pos][plot_id].data[data_id]["data_vec"]
        [R, G, B] = self.plots_vec[pos][plot_id].data[data_id]["color"]
        label = self.plots_vec[pos][plot_id].data[data_id]["label"]
        if (pos == "Upper") or (pos == "Lower"):
            self.plots_vec[pos][plot_id].data_plot[data_id] = self.plots_vec[pos][plot_id].plot.plot(x = self.t0, pen = [R,G,B],
                                                                                                name = label)
            self.plots_vec[pos][plot_id].data_plot[data_id].setData(data_time, data_vec)
            if self.video_created:
                self.plots_vec[pos][plot_id].plot.setXRange(self.time_vid-self.plot_interval, self.time_vid, padding=0)
    def set_data_info(self, data):
        info_text = "- Type: " + str(type(data)) + "\n"
        if type(data) is np.ndarray:
            info_text += "- Shape: "
            info_text += str(data.shape)
            info_text += "\n"
            info_text += "- Data: \n"
            info_text += str(data)
        elif type(data) is list:
            info_text += "- Len: "
            info_text += str(len(data))
            info_text += "\n"
            info_text += "- Data: \n"
            info_text += str(data)
        elif type(data) is dict:
            info_text += "- Keys: "
            info_text += str([*data])
            info_text += "\n"
            info_text += "- Data: \n"
            for key in [*data]:
                info_text += str(key) + ": "
                info_text += str(data[key]) + "\n"
        else:
            info_text += "- Data: \n"
            info_text += str(data)
        self.ui_dw.data_info.setPlainText(info_text)
    def update_data_list(self, plot_id):
        pos = self.find_pos_from_id(plot_id)
        self.ui_pw.data_list.clear()
        self.ui_pw.data_list.addItems(self.plots_vec[pos][plot_id].activeData())
    def get_data_from_data_line(self, data_line):
        data_line_tmp = data_line
        for data_id in self.selected_data_dic.keys():
            if data_id in data_line:
                data_line_tmp = data_line_tmp.replace(data_id, "self.selected_data_dic['{}']".format(data_id))
        for function_id in self.opened_functions_path.keys():
            if function_id in data_line:
                file_path = self.opened_functions_path[function_id]
                self.opened_functions[function_id] = function_from_file(file_path, file_path.split('/')[-1].split('.')[0])
                data_line_tmp = data_line_tmp.replace(function_id, "self.opened_functions['{}']".format(function_id))
        print("Evaluating: ", data_line_tmp)
        [data_vec, data_time] = eval(data_line_tmp)
        return data_vec, data_time
# General Functions -----------------------------------------------------------------------------------------------------------------
    def find_pos_from_id(self, plot_id):
        pos = ""
        print("Find pos from ID")
        print("Plot ID: ", plot_id)
        for key in self.plots_vec.keys():
            if plot_id in self.plots_vec[key].keys():
                pos = key
        print("Pos: ", pos)
        if pos == "":
            print("Plot not found")
            return
        return pos
    def file_open_dialog(self):
        file_path = self.readFileDialog(title="Open State File")
        try:
            data_dic, var_names = parse_pickle_file(file_path)
            # Delete current plots
            for pos in self.plots_vec.keys():
                if pos != "Inactive":
                    for plot_id in self.plots_vec[pos].keys():
                        self.window_plots[pos].removeItem(self.plots_vec[pos][plot_id].plot)
            
            self.opened_files = data_dic["opened_files"]
            self.selected_data_dic = data_dic["selected_data_dic"]
            self.opened_functions_path = data_dic["opened_functions_path"]
            self.opened_functions = data_dic["opened_functions"]
            self.ui_fw.files_list_wg.clear()
            self.ui_fw.files_list_wg.addItems([*self.opened_files])
            self.ui_fw.selected_data.clear()
            self.ui_fw.selected_data.addItems([*self.selected_data_dic])
            self.ui_dw.opened_functions.clear()
            self.ui_dw.opened_functions.addItems([*self.opened_functions_path])

            self.plots_vec = data_dic["plots_vec"]
            if (pos == "Upper") or (pos == "Lower"):
                if pos != "Inactive":
                    for plot_id in self.plots_vec[pos].keys():
                        self.plots_vec[pos][plot_id].plot = self.window_plots[pos].addPlot(title=self.plots_vec[pos][plot_id].title,
                                                                    labels = {'left':self.plots_vec[pos][plot_id].y_label, 
                                                                            'bottom':self.plots_vec[pos][plot_id].x_label})
                        self.plots_vec[pos][plot_id].plot.addLegend()
                        self.set_data_plot(plot_id)
            self.update_plot_list()
        except Exception as e:
            print("error: ", e)
        
    def save_dialog(self):
        if self.save_file_path == "":
            self.save_file_path = self.saveFileDialog(title="Save file as")
        self.store_all_data()
    def save_as_dialog(self):
        self.save_file_path = self.saveFileDialog(title="Save file as")
        self.store_all_data()
    def store_all_data(self):
        data_dic = {
            "selected_data_dic": self.selected_data_dic,
            "opened_files": self.opened_files,
            "opened_functions_path": self.opened_functions_path,
            "opened_functions": self.opened_functions,
            "plots_vec": self.plots_vec
        }
        save_to_file(data_dic, self.save_file_path)
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
    def saveFileDialog(self, title="Save File"):
        self.options = QtWidgets.QFileDialog.Options()
        self.options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name = QtWidgets.QFileDialog.getSaveFileName(None, title, options=self.options)
        return file_name

if __name__ == "__main__":
    MainWindow_EXEC()