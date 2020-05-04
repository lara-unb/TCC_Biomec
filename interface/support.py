import numpy as np
import pickle
from data_processing import *
from data_classification import *
from scipy.signal import medfilt
from sklearn.preprocessing import StandardScaler
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTime, QTimer, QSize
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget, QTableWidgetItem, QVBoxLayout, QFileDialog

def parse_pickle_file(filename):
    data = {}
    # Load data
    with open(filename, 'rb') as f:
        try:
            while True:
                data.update({pickle.load(f): pickle.load(f)})
    
        except EOFError:
            pass
    
    var_names = []
    for k, v in data.items():
        var_names.append(k)

    return data, var_names

def get_stim_value(stim_state):
    if stim_state.find('stop') != -1:
        return 0
    elif stim_state.find('extension') != -1:
        return 1
    elif stim_state.find('flexion') != -1:
        return -1

def get_starting_time(filenames):
    times = []
    for filename in filenames:
        with open(filename) as inputfile:
            for line in inputfile:
                line = line.split(',')
                times.append(float(line[0]))
                break

    return min(times)

def get_imus_resampled_data(imus, imu_forearm_id, imu_arm_id, signal, pos):
    if imus[0].id == imu_forearm_id:
        imu_0 = 0
        imu_1 = 1
    else:
        imu_1 = 0
        imu_0 = 1
    if pos == "forearm":
        imu_pos = imu_0
    else:
        imu_pos = imu_1
    if signal == "x":
        return imus[imu_pos].resampled_x
    elif signal == "y":
        return imus[imu_pos].resampled_y
    elif signal == "z":
        return imus[imu_pos].resampled_z
    elif signal == "w":
        return imus[imu_pos].resampled_w
    elif signal == "acc_x":
        return imus[imu_pos].resampled_acc_x
    elif signal == "acc_y":
        return imus[imu_pos].resampled_acc_y
    else:
        return imus[imu_pos].resampled_acc_z

def parse_imus_file(filename, imu_forearm_id, imu_arm_id, starting_time=0, i_lim=0, s_lim=0, crop=False):
    lines = []
    imus = []
    imus_ids = []
    with open(filename) as inputfile:
        for line in inputfile:
            lines.append(line.split(','))
    if crop:
        for data in lines[1:]:
            if float(data[0]) - starting_time >= i_lim:
                if float(data[0]) - starting_time > s_lim:
                    break
                id = float(data[2])
                if id not in imus_ids:
                    imus_ids.append(id)
                    imus.append(IMU(id))
                imus[imus_ids.index(id)].timestamp.append(float(data[0]) - starting_time)
                imus[imus_ids.index(id)].x_values.append(float(data[3]))
                imus[imus_ids.index(id)].y_values.append(float(data[4]))
                imus[imus_ids.index(id)].z_values.append(float(data[5]))
                imus[imus_ids.index(id)].w_values.append(float(data[6]))
                imus[imus_ids.index(id)].acc_x.append(float(data[7]))
                imus[imus_ids.index(id)].acc_y.append(float(data[8]))
                imus[imus_ids.index(id)].acc_z.append(float(data[9]))
    else:
        for data in lines[1:]:
            if float(data[0]) >= starting_time:
                id = float(data[2])
                if id not in imus_ids:
                    imus_ids.append(id)
                    imus.append(IMU(id))
                imus[imus_ids.index(id)].timestamp.append(float(data[0]) - starting_time)
                imus[imus_ids.index(id)].x_values.append(float(data[3]))
                imus[imus_ids.index(id)].y_values.append(float(data[4]))
                imus[imus_ids.index(id)].z_values.append(float(data[5]))
                imus[imus_ids.index(id)].w_values.append(float(data[6]))
                imus[imus_ids.index(id)].acc_x.append(float(data[7]))
                imus[imus_ids.index(id)].acc_y.append(float(data[8]))
                imus[imus_ids.index(id)].acc_z.append(float(data[9]))

    [imus[i].get_euler_angles() for i in range(len(imus))]
    if imus[0].id == imu_forearm_id:
        imu_0 = 0
        imu_1 = 1
    else:
        imu_1 = 0
        imu_0 = 1

    print('Resampling and synchronizing...')

    [t, imus[imu_0].resampled_x, imus[imu_1].resampled_x] = resample_series(imus[imu_0].timestamp,
                                                                            imus[imu_0].x_values,
                                                                            imus[imu_1].timestamp,
                                                                            imus[imu_1].x_values)
    [t, imus[imu_0].resampled_y, imus[imu_1].resampled_y] = resample_series(imus[imu_0].timestamp,
                                                                            imus[imu_0].y_values,
                                                                            imus[imu_1].timestamp,
                                                                            imus[imu_1].y_values)
    [t, imus[imu_0].resampled_z, imus[imu_1].resampled_z] = resample_series(imus[imu_0].timestamp,
                                                                            imus[imu_0].z_values,
                                                                            imus[imu_1].timestamp,
                                                                            imus[imu_1].z_values)
    [t, imus[imu_0].resampled_w, imus[imu_1].resampled_w] = resample_series(imus[imu_0].timestamp,
                                                                            imus[imu_0].w_values,
                                                                            imus[imu_1].timestamp,
                                                                            imus[imu_1].w_values)
    [t, imus[imu_0].resampled_acc_x, imus[imu_1].resampled_acc_x] = resample_series(imus[imu_0].timestamp,
                                                                                    imus[imu_0].acc_x,
                                                                                    imus[imu_1].timestamp,
                                                                                    imus[imu_1].acc_x)
    [t, imus[imu_0].resampled_acc_y, imus[imu_1].resampled_acc_y] = resample_series(imus[imu_0].timestamp,
                                                                                    imus[imu_0].acc_y,
                                                                                    imus[imu_1].timestamp,
                                                                                    imus[imu_1].acc_y)
    [t, imus[imu_0].resampled_acc_z, imus[imu_1].resampled_acc_z] = resample_series(imus[imu_0].timestamp,
                                                                                    imus[imu_0].acc_z,
                                                                                    imus[imu_1].timestamp,
                                                                                    imus[imu_1].acc_z)

    return imus, t

# Find files with EMG, IMU and button data
def separate_files(filenames):
    emg_files = [f for f in filenames if 'EMG' in f]
    imus_files = [f for f in filenames if 'imu' in f]
    buttons_files = [f for f in filenames if 'stim' in f]
    pos_files = [f for f in filenames if 'pos' in f]
    vidtime_files = [f for f in filenames if 'vidtime' in f]
    video_files = [f for f in filenames if 'tracking' in f]
    return [emg_files, imus_files, buttons_files, pos_files, vidtime_files, video_files]

def update_classes_and_transitions(classes, transitions, classes_new, transitions_new):
    classes = list(set(classes + classes_new))
    for item in transitions_new:
        if item not in transitions:
            transitions += transitions_new
    return classes, transitions

def find_classes_and_transitions(labels):
    classes = []
    transitions = []
    previous_label = []
    for label in labels:
        if label not in classes:
            classes.append(label)
        if (label != previous_label) and (previous_label != []):
            if [previous_label, label] not in transitions:
                transitions.append([previous_label, label])
        previous_label = label

    return classes, transitions

def parse_pos_file(pos_file, vidtime_file, starting_time=0, i_lim=0, s_lim=0, crop=False):
    pos_data = np.loadtxt(pos_file, delimiter=',')[1:, 0]
    video_time = np.loadtxt(vidtime_file)
    pos_data_out = []
    pos_time = []
    if crop:
        for i in range(len(video_time)):
            if float(video_time[i]) - starting_time >= i_lim:
                if float(video_time[i]) - starting_time > s_lim:
                    break
                pos_time.append(float(video_time[i]) - starting_time)
                pos_data_out.append(pos_data[i])
    else:
        for i in range(len(video_time)):
            if float(video_time[i]) >= starting_time:
                pos_time.append(float(video_time[i]) - starting_time)
                pos_data_out.append(pos_data[i])
    return [pos_time, pos_data_out]

def parse_stim_file(filename, starting_time=0, i_lim=0, s_lim=0, crop=False):
    lines = []
    timestamp = []
    stim_state = []
    with open(filename) as inputfile:
        for line in inputfile:
            lines.append(line.split(','))
    if crop:
        for i in range(len(lines)):
            if float(lines[i][0]) - starting_time >= i_lim:
                if float(lines[i][0]) - starting_time > s_lim:
                    break
                timestamp.append(float(lines[i][0]) - starting_time)
                stim_state.append(get_stim_value(lines[i][2]))
    else:
        for i in range(len(lines)):
            if float(lines[i][0]) >= starting_time:
                timestamp.append(float(lines[i][0]) - starting_time)
                stim_state.append(get_stim_value(lines[i][2]))
    return [timestamp, stim_state]

def parse_out_file(filename):
    data = {}
    # Load data
    print('Loading data from file {}'.format(filename))
    with open(filename, 'rb') as f:
        try:
            while True:
                data.update({pickle.load(f): pickle.load(f)})
    
        except EOFError:
            print('Loading complete')
    
    var_names = []
    for k, v in data.items():
        var_names.append(k)
    
    print('Variables loaded: ', var_names)
    return data, var_names

class GetFolderToLoad(QWidget):

    def __init__(self):
        super(GetFolderToLoad, self).__init__()
        self.foldername = []
        self.openFileDialog()

    def openFileDialog(self):
        foldername = QFileDialog.getExistingDirectory(self)
        if foldername:
            self.foldername = foldername


class GetFileToSave(QWidget):

    def __init__(self):
        super(GetFileToSave, self).__init__()
        self.filename = []
        self.openFileDialog()

    def openFileDialog(self):
        filename = QFileDialog.getSaveFileName(self)
        if filename:
            self.filename = filename


class GetFilesToLoad(QWidget):

    def __init__(self):
        super(GetFilesToLoad, self).__init__()
        self.filename = []
        self.openFileDialog()

    def openFileDialog(self):
        filename = QFileDialog.getOpenFileNames(self)
        if filename:
            self.filename = filename

def saveFileDialog():
    app = QApplication(sys.argv)
    output_path, _ = QFileDialog.getSaveFileName()
    return output_path

def readFileDialog(title, file_type="All Files"):
    app = QApplication(sys.argv)
    qfd = QFileDialog()
    if file_type == "All Files":
        type_filter = "All Files (*)"
    else:
        type_filter = file_type + " (*." + file_type + ")"
    file_path, _ = QFileDialog.getOpenFileName(qfd, title, "", type_filter)
    return file_path

def readFilesDialog(title, file_type="All Files"):
    app = QApplication(sys.argv)
    qfd = QFileDialog()
    if file_type == "All Files":
        type_filter = "All Files (*)"
    else:
        type_filter = file_type + " (*." + file_type + ")"
    file_path, _ = QFileDialog.getOpenFileNames(qfd, title, "", type_filter)
    return file_path

def save_to_file(data, filename, data_str=[]):
    i = 0
    with open(filename, 'wb') as f:
        for piece_of_data in data:
            if len(data_str) == 0:
                pickle.dump(str(i), f)
                pickle.dump(piece_of_data, f)
            else:
                pickle.dump(data_str[i], f)
                pickle.dump(piece_of_data, f)
            i+=1

def append_features(X_vec, y_vec, X_new, y_new):
    X_vec = X_vec + X_new
    y_vec = y_vec + y_new
    return X_vec, y_vec

def get_file_features(filename):
    data, feat_names = parse_out_file(filename)
    X = []
    y = data["stim_label"]
    t = data["t"]
    for i in range(len(t)):
        X_tmp = []
        for feature in feat_names[:-2]:    
            X_tmp.append(data[feature][i])
        X.append(X_tmp)
            
    return X, y, t

def multi_clf_separation(X, y, classes, transitions, crop=True):
    X_out = []
    y_out = []
    for transition in transitions:
        if crop:
            X_tmp = X[(y==transition[0]) | (y==transition[1])]
            y_tmp = y[(y==transition[0]) | (y==transition[1])]
            X_out.append(np.array(X_tmp))
            y_out.append(np.array(y_tmp))
        else:
            for label in classes:
                if label not in transition:
                    y_tmp = np.where(y==label, transition[1], y)
            X_out.append(np.copy(X))
            y_out.append(y_tmp)
    return X_out, y_out