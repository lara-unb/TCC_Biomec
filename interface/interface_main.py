from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTime, QTimer, QSize
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget, QTableWidgetItem, QVBoxLayout, QDialog
import pyqtgraph as pg
import numpy as np
import sys
import cv2
import time
from time import sleep
import pyqtgraph as pg
from processing_interface import Ui_MainWindow
import matplotlib.pyplot as plt
from data_processing import generate_imu_data, scale_features
from support import parse_imus_file, parse_stim_file, get_imus_resampled_data, parse_out_file
import importlib

#TODO: define Feature extractor model; config Feat and Pred graphs, apply PCA, organize repo

class myPlot():
    def __init__(self, num, title, y_label, x_label, segment=False):
        self.num = num
        self.title = title
        self.y_label = y_label
        self.x_label = x_label
        self.data = {"Pos": {}, "IMU": {}, "Stim": {}, "Out": {}, "Feat": {}, "Pred": {}}
        self.segment = segment
    def activeData(self):
        return self.data.keys()
    def removeData(self, file_name, element):
        self.data[file_name].remove(element)

class MainWindow_EXEC():
    def __init__(self):
        app = QtWidgets.QApplication(sys.argv)
        
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)   

        self.frame_n = self.ui.frame_n.value()
        self.ui.frame_slider.setProperty("value", self.frame_n)

        self.sync_time = 0

        self.ui.browse_imu.clicked.connect(self.on_browse_imu_clicked)
        self.ui.browse_stim.clicked.connect(self.on_browse_stim_clicked)
        self.ui.browse_pos.clicked.connect(self.on_browse_pos_clicked)
        self.ui.browse_video.clicked.connect(self.on_browse_video_clicked)
        self.ui.browse_vidtime.clicked.connect(self.on_browse_vidtime_clicked)
        self.ui.browse_out.clicked.connect(self.on_browse_out_clicked)
        self.ui.browse_feat.clicked.connect(self.on_browse_feat_clicked)
        self.ui.browse_pred.clicked.connect(self.on_browse_pred_clicked)
        self.ui.browse_processing.clicked.connect(self.on_browse_processing_clicked)
        self.ui.pp_button.clicked.connect(self.on_pp_button_clicked)
        self.ui.speed_n.valueChanged.connect(self.on_speed_n_changed)
        self.ui.frame_n.valueChanged.connect(self.on_frame_n_changed)
        self.ui.frame_slider.valueChanged.connect(self.on_frame_slider_changed)
        self.ui.ca_plot_button.clicked.connect(self.on_ca_plot_button_clicked)
        self.ui.rm_plot_button.clicked.connect(self.on_rm_plot_button_clicked)
        self.ui.file_data_set.currentTextChanged.connect(self.on_file_data_set_changed)
        self.ui.add_data_button.clicked.connect(self.on_add_data_button_clicked)
        self.ui.rm_data_button.clicked.connect(self.on_rm_data_button_clicked)
        self.ui.imu_path.textChanged.connect(self.on_imu_path_changed)
        self.ui.stim_path.textChanged.connect(self.on_stim_path_changed)
        self.ui.pos_path.textChanged.connect(self.on_pos_path_changed)
        self.ui.vid_path.textChanged.connect(self.on_vid_path_changed)
        self.ui.vidtime_path.textChanged.connect(self.on_vidtime_path_changed)
        self.ui.out_path.textChanged.connect(self.on_out_path_changed)  
        self.ui.interval.valueChanged.connect(self.on_interval_changed)
        self.ui.feat_path.textChanged.connect(self.on_feat_path_changed)

        self.load_files()

        self.size_of_graph = 1000
        self.t0 = np.zeros(self.size_of_graph)

        self.max_plots = 4
        self.window_plots = {"Upper": self.ui.window_plot_upper, "Lower": self.ui.window_plot_lower}
        self.plots_vec = {"Upper": {"num":[]}, "Lower": {"num":[]}}

        self.MainWindow.show()
        sys.exit(app.exec_())

    def load_files(self):
        self.on_vid_path_changed()
        self.on_vidtime_path_changed()
        self.on_imu_path_changed()
        self.on_stim_path_changed()
        self.on_pos_path_changed()
        self.on_out_path_changed()

    def on_vid_path_changed(self):
        self.video_size=QSize(640,480)
        self.image=None
        self.processedImage=None
        self.video_path = self.ui.vid_path.text()
        print("Speed: {}".format(self.ui.speed_n.value()))

    def on_stim_path_changed(self):
        try:
            if self.ui.stim_path.text() != "":
                [self.time_stim, self.data_stim] = parse_stim_file(self.ui.stim_path.text())
        except:
            print("Can't open stim file")

    def on_imu_path_changed(self):
        try:
            if self.ui.imu_path.text() != "":
                self.imus, self.time_imu = parse_imus_file(self.ui.imu_path.text(), 
                                                            self.ui.forearm_id.value(), 
                                                            self.ui.arm_id.value())
                self.qang, self.dqang, _ = generate_imu_data(self.imus, self.time_imu, 
                                                            self.ui.forearm_id.value(), 
                                                            self.ui.arm_id.value())
        except:
            print("Can't open imu file")

    def on_pos_path_changed(self):
        try:
            if self.ui.pos_path.text() != "":
                self.data_pos = np.loadtxt(self.ui.pos_path.text(), delimiter=',')[1:, 0]
        except:
            print("Can't open pos file")

    def on_vidtime_path_changed(self):
        try:
            if self.ui.vidtime_path.text() != "":
                self.data_time = np.loadtxt(self.ui.vidtime_path.text())
                self.sync_time = self.data_time[0]
                self.data_time -= self.sync_time
                self.time_vid = self.data_time[self.frame_n]
                self.fps = int(1/np.mean(np.diff(self.data_time)))
        except:
            print("Can't open vidtime file")
    
    def on_out_path_changed(self):
        if self.ui.out_path.text() != "":
            data, var_names = parse_out_file(self.ui.out_path.text())
            self.sync_time = data['absolute_time']
            self.out_stim_data = data['stim_values']
            self.out_stim_time = data['stim_timestamp']
            self.out_imus = data['imus']
            self.out_imu_time = data['t']
            self.out_imu_forearm_id = data['imu_forearm_id']
            self.out_imu_arm_id = data['imu_arm_id']
            self.out_qang, self.out_dqang, _ = generate_imu_data(self.out_imus, self.out_imu_time, 
                                                        self.out_imu_forearm_id, self.out_imu_arm_id)
            self.out_pos_data = data['pos_data']
            self.out_pos_time = data['pos_time']
            self.time_vid = self.out_pos_time[self.frame_n]
            self.fps = int(1/np.mean(np.diff(self.out_pos_time)))
    
    def on_feat_path_changed(self):
        if self.ui.feat_path.text() != "":
            data, self.feat_names = parse_out_file(self.ui.feat_path.text())
            self.feat_vec = {}
            for feature in self.feat_names:    
                self.feat_vec[feature] = data[feature]
            self.feat_time = data["t"]

    def on_file_data_set_changed(self):
        self.ui.elem_data_set.clear()
        if self.ui.file_data_set.currentText() == "IMU":
            str_list = ["qang", "dqang", "x forearm", "y forearm", "z forearm", "w forearm", 
                        "acc_x forearm", "acc_y forearm", "acc_z forearm", "x arm", "y arm", 
                        "z arm", "w arm", "acc_x arm", "acc_y arm", "acc_z arm"]
            self.ui.elem_data_set.insertItems(0, str_list)
        elif self.ui.file_data_set.currentText() == "Stim":
            str_list = ["stim app"]
            self.ui.elem_data_set.insertItems(0, str_list)
        elif self.ui.file_data_set.currentText() == "Pos":
            str_list = ["seat position"]
            self.ui.elem_data_set.insertItems(0, str_list)
        if self.ui.file_data_set.currentText() == "Out":
            str_list = ["qang", "dqang", "stim_app", "pos", "x forearm", "y forearm", 
                        "z forearm", "w forearm", "acc_x forearm", "acc_y forearm", "acc_z forearm",
                        "x arm", "y arm", "z arm", "w arm", "acc_x arm", "acc_y arm", "acc_z arm"]
            self.ui.elem_data_set.insertItems(0, str_list)
        if self.ui.file_data_set.currentText() == "Feat":
            self.ui.elem_data_set.insertItems(0, self.feat_names)

    def on_add_data_button_clicked(self):
        n = self.ui.data_n_set.value()
        pos = self.ui.data_set_ul.currentText()
        element = self.ui.elem_data_set.currentText()
        file_name = self.ui.file_data_set.currentText()
        sync = False
        if element in self.plots_vec[pos][n].data[file_name].keys():
            self.plots_vec[pos][n].plot.removeItem(self.plots_vec[pos][n].data[file_name][element])

        if file_name == "IMU":
            if element == "qang":
                data_vec = np.copy(self.qang)
            elif element == "dqang":
                data_vec = np.copy(self.dqang)
            else:
                elem = element.split(" ")[0]
                elem_pos = element.split(" ")[-1]
                data_vec = get_imus_resampled_data(self.imus, self.ui.forearm_id.value(), 
                                                    self.ui.arm_id.value(), elem, elem_pos)
            data_time = np.copy(self.time_imu)
            sync = True
        if file_name == "Stim":
            data_vec = np.copy(self.data_stim)
            data_time = np.copy(self.time_stim)
            sync = True
        if file_name == "Pos":
            data_vec = np.copy(self.data_pos)
            data_time = np.copy(self.data_time)
            sync = False
        if file_name == "Out":
            if element == "qang":
                data_vec = np.copy(self.out_qang)
                data_time = np.copy(self.out_imu_time)
                sync = False
            elif element == "dqang":
                data_vec = np.copy(self.out_dqang)
                data_time = np.copy(self.out_imu_time)
                sync = False
            elif element == "stim_app":
                data_vec = np.copy(self.out_stim_data)
                data_time = np.copy(self.out_stim_time)
                sync = False
            elif element == "pos":
                data_vec = np.copy(self.out_pos_data)
                data_time = np.copy(self.out_pos_time)
                sync = False
            else:
                elem = element.split(" ")[0]
                elem_pos = element.split(" ")[-1]
                data_vec = get_imus_resampled_data(self.out_imus, self.out_imu_forearm_id, 
                                                self.out_imu_arm_id, elem, elem_pos)
                data_time = np.copy(self.out_imu_time)
                sync = False
        if file_name == "Feat":
            data_vec = self.feat_vec[element]
            data_time = np.copy(self.feat_time)
            sync = False

        if sync:
            data_time -= self.sync_time

        if self.ui.apply_processing.isChecked():
            data_vec = np.array(data_vec)
            processing_file_ext = self.ui.processing_function.text()
            file_parts = processing_file_ext.split(sep='/')
            processing_file_dir = ""
            for i in range(len(file_parts) - 1):
                processing_file_dir += file_parts[i] + "/" 
            sys.path.append(processing_file_dir)
            processing_file = file_parts[-1].split(sep='.')[0]

            processing = __import__(processing_file)
            importlib.reload(sys.modules[processing_file])
            processing = __import__(processing_file)
            processing_function = getattr(processing, 'processing_function')
            data_vec, data_time = processing_function(data_vec, data_time)
            element = element + "_" + self.ui.label_data_set.text()

        if self.ui.scale_cb.isChecked():
            data_vec = np.array(scale_features(data_vec, np.mean(data_vec), np.std(data_vec), single=True))

        R, G, B = self.ui.data_R_n.value(), self.ui.data_G_n.value(), self.ui.data_B_n.value()
        self.plots_vec[pos][n].data[file_name][element] = self.plots_vec[pos][n].plot.plot(x = self.t0, pen = [R,G,B],
                                                                                            name = self.ui.label_data_set.text())
        self.plots_vec[pos][n].data[file_name][element].setData(data_time, data_vec)
        self.plots_vec[pos][n].plot.setXRange(self.time_vid-self.ui.interval.value(), self.time_vid, padding=0)

    def on_rm_data_button_clicked(self):
        n = self.ui.data_n_set.value()
        pos = self.ui.data_set_ul.currentText()
        element = self.ui.elem_data_set.currentText()
        file_name = self.ui.file_data_set.currentText()
        if self.ui.apply_processing.isChecked():
            element = element + "_" + self.ui.label_data_set.text()
        self.plots_vec[pos][n].plot.removeItem(self.plots_vec[pos][n].data[file_name][element])
        del self.plots_vec[pos][n].data[file_name][element]

    def on_ca_plot_button_clicked(self):
        n = self.ui.plot_n_set.value()
        pos = self.ui.plt_set_ul.currentText()
        if n not in self.plots_vec[pos]["num"]:
            self.plots_vec[pos]["num"].append(n)
            print("Creating graph")
        else:
            print("Altering graph")
            self.window_plots[pos].removeItem(self.plots_vec[pos][n].plot)
            
        self.plots_vec[pos][n] = myPlot(num = n,
                                        title = self.ui.title_plot.text(),
                                        y_label = self.ui.y_label.text(),
                                        x_label = self.ui.x_label.text())
        self.plots_vec[pos][n].plot = self.window_plots[pos].addPlot(title=self.plots_vec[pos][n].title,
                                                            labels = {'left':self.plots_vec[pos][n].y_label, 
                                                                    'bottom':self.plots_vec[pos][n].x_label})
        self.plots_vec[pos][n].plot.addLegend()
    
    def on_rm_plot_button_clicked(self):
        pos = self.ui.plt_set_ul.currentText()
        try:
            n = self.ui.plot_n_set.value()
            self.window_plots[pos].removeItem(self.plots_vec[pos][n].plot)
            self.plots_vec[pos]["num"].remove(n)
            self.plots_vec[pos].remove(n)
        except:
            print("Plot number invalid")

    def format_seconds_to_mmss(self, seconds):
        minutes = seconds // 60
        seconds %= 60
        return "%02i:%02i" % (minutes, seconds)

    def on_frame_slider_changed(self):
        self.ui.frame_n.valueChanged.disconnect()
        self.frame_n = self.ui.frame_slider.value()
        self.ui.frame_n.setProperty("value", self.frame_n)
        self.update_frame()
        print("New frame: {}".format(self.frame_n))
        self.ui.frame_n.valueChanged.connect(self.on_frame_n_changed)

    def on_frame_n_changed(self):
        self.frame_n = self.ui.frame_n.value()
        self.ui.frame_slider.setProperty("value", self.frame_n)
    
    def on_interval_changed(self):
        self.update_plot_interval()

    def on_speed_n_changed(self):
        try:
            print("Speed: {}".format(self.ui.speed_n.value()))
            self.timer.stop()
            self.timer.start(int(1000/(self.fps*self.ui.speed_n.value())))
        except:
            pass

    def on_browse_imu_clicked(self):
        print("Browse clicked")
        imu_path = self.readFileDialog()
        self.ui.imu_path.setText(imu_path)
        print(imu_path)
    
    def on_browse_stim_clicked(self):
        print("Browse clicked")
        stim_path = self.readFileDialog()
        self.ui.stim_path.setText(stim_path)
        print(stim_path)
    
    def on_browse_pos_clicked(self):
        print("Browse clicked")
        pos_path = self.readFileDialog()
        self.ui.pos_path.setText(pos_path)
        print(pos_path)

    def on_browse_video_clicked(self):
        print("Browse clicked")
        video_path = self.readFileDialog()
        self.ui.vid_path.setText(video_path)
        print(video_path)
    
    def on_browse_vidtime_clicked(self):
        print("Browse clicked")
        vidtime_path = self.readFileDialog()
        self.ui.vidtime_path.setText(vidtime_path)
        print(vidtime_path)
    
    def on_browse_out_clicked(self):
        print("Browse clicked")
        out_path = self.readFileDialog()
        self.ui.out_path.setText(out_path)
        print(out_path)
    
    def on_browse_feat_clicked(self):
        print("Browse clicked")
        feat_path = self.readFileDialog()
        self.ui.feat_path.setText(feat_path)
        print(feat_path)
    
    def on_browse_pred_clicked(self):
        print("Browse clicked")
        pred_path = self.readFileDialog()
        self.ui.pred_path.setText(pred_path)
        print(pred_path)

    def on_browse_processing_clicked(self):
        print("Browse clicked")
        processing_path = self.readFileDialog()
        self.ui.processing_function.setText(processing_path)
        print(processing_path)

    def on_pp_button_clicked(self):
        print("Play/Pause button clicked")
        if self.ui.pp_button.text() == "Play":
            if self.video_path == "":
                print("Select a video file")
            else:
                self.ui.pp_button.setText("Pause")
                self.ui.icon = QtGui.QIcon()
                self.ui.icon.addPixmap(QtGui.QPixmap("pause.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                self.ui.pp_button.setIcon(self.ui.icon)
                self.ui.frame_n.valueChanged.disconnect()
                self.ui.frame_slider.valueChanged.disconnect()
                self.start_webcam()
        else:
            self.ui.pp_button.setText("Play")
            self.ui.icon = QtGui.QIcon()
            self.ui.icon.addPixmap(QtGui.QPixmap("play.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.ui.pp_button.setIcon(self.ui.icon)
            self.ui.frame_n.valueChanged.connect(self.on_frame_n_changed)
            self.ui.frame_slider.valueChanged.connect(self.on_frame_slider_changed)
            self.stop_webcam()

    def start_webcam(self):
        self.capture=cv2.VideoCapture(self.video_path)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,self.video_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,self.video_size.height())
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.fps = 30
        print("FPS: {}".format(self.fps))
        self.frame_total = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.ui.frame_n.setProperty("maximum", self.frame_total)
        self.ui.frame_slider.setProperty("maximum", self.frame_total)
        self.time_max = self.frame_total / self.fps
        self.time_max_text = self.format_seconds_to_mmss(self.time_max)
        print("Number of frames: {}".format(self.frame_total))

        self.capture.set(1, self.frame_n)

        self.timer = QTimer()
        self.timer.timeout.connect(self.play_video)
        self.timer.start(int(1000/(self.fps*self.ui.speed_n.value())))

    def play_video(self):
        self.ui.frame_n.setProperty("value", self.frame_n)
        self.ui.frame_slider.setProperty("value", self.frame_n)
        self.update_frame()
        self.frame_n += 1

    def update_frame(self):
        try:
            ret,frame=self.capture.read()
            self.image=cv2.cvtColor(frame,1)
            self.processedImage=self.image
            self.displayImage(1)
            try:
                self.time_vid = self.data_time[self.frame_n]
            except:
                try:
                    self.time_vid = self.out_pos_time[self.frame_n]
                except:
                    self.time_vid = (1/self.fps) * self.frame_n
            time_text = self.format_seconds_to_mmss(self.time_vid)
            time_text = time_text + " / " + self.time_max_text
            self.ui.time_video.setText(time_text)
            self.update_plot_interval()
        except:
            pass

    def update_plot_interval(self):
        for i in self.plots_vec["Upper"]["num"]:
            self.plots_vec["Upper"][i].plot.setXRange(self.time_vid-self.ui.interval.value(), self.time_vid, padding=0)
        for i in self.plots_vec["Lower"]["num"]:
            self.plots_vec["Lower"][i].plot.setXRange(self.time_vid-self.ui.interval.value(), self.time_vid, padding=0)

    def stop_webcam(self):
        self.timer.stop()

    def displayImage(self, window=1):
        qformat = QImage.Format_Indexed8

        if len(self.processedImage.shape) == 3:  # rows[0],cols[1],channels[2]
            if (self.processedImage.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        print(self.processedImage.shape)
        width = 640
        height = 480
        dim = (width, height)
        self.processedImage = cv2.resize(self.processedImage, dim, interpolation = cv2.INTER_AREA)
        img = QImage(self.processedImage, width, height,
                    self.processedImage.strides[0], qformat)
        # BGR > RGB
        img = img.rgbSwapped()
        if window == 1:
            self.ui.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.ui.imgLabel.setScaledContents(True)
    
    def readFileDialog(self):
        self.options = QtWidgets.QFileDialog.Options()
        self.options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()",
                                                            "","All Files (*);;Text Files (*.txt)", 
                                                            options=self.options)
        return file_name

if __name__ == "__main__":
    MainWindow_EXEC()