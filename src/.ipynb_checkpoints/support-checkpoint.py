import cv2
import time
import numpy as np
import json
import math
import sys
import os
import pickle
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtWidgets import QApplication
from parameters import *
from kinematics import inverseKinematicsRowing
import importlib

def save_to_file(data_dic, file_path):
    with open(file_path, 'wb') as f:
        for key in data_dic.keys():
            pickle.dump(key, f)
            pickle.dump(data_dic[key], f)

def function_from_file(file_path, function_name):
    folder_path = ""
    for string in file_path.split("/")[:-1]:
        folder_path += string + "/"
    file_name = file_path.split("/")[-1].split(".")[0]
    if folder_path not in sys.path:
        sys.path.append(folder_path)
    processing = __import__(file_name)
    importlib.reload(sys.modules[file_name])
    processing = __import__(file_name)
    processing_function = getattr(processing, function_name)
    return processing_function

def saveDATAtoFile(file_path, data):
    writeToDATA(file_path, data["metadata"], write_mode='w')
    for i in range(len(data["keypoints"])):
        pose_keypoints = data["keypoints"][i]
        try:
            angles = data["angles"][i]
        except:
            angles = []
        file_data = {
                    'keypoints': pose_keypoints.tolist(),
                    'angles': angles.tolist()
                    }
        writeToDATA(file_path, file_data, write_mode='a')

def readFileDialog(title="Open File", file_type="All Files"):
    app = QApplication(sys.argv)
    qfd = QFileDialog()
    if file_type == "All Files":
        type_filter = "All Files (*)"
    else:
        type_filter = file_type + " (*." + file_type + ")"
    file_path, _ = QFileDialog.getOpenFileName(qfd, title, "", type_filter)
    return file_path

def readFolderDialog(title="Open Folder"):
    app = QApplication(sys.argv)
    qfd = QFileDialog()
    folder_path = QFileDialog.getExistingDirectory(qfd, title)
    return folder_path

def saveFileDialog(title="Save File"):
    app = QApplication(sys.argv)
    qfd = QFileDialog()
    file_path = QFileDialog.getSaveFileName(qfd, title)
    return file_path[0]

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

class GetFolderToLoad(QWidget):

    def __init__(self):
        super(GetFolderToLoad, self).__init__()
        self.foldername = []
        self.openFileDialog()

    def openFileDialog(self):
        # home = os.getenv("HOME")
        foldername = QFileDialog.getExistingDirectory(self)
        if foldername:
            self.foldername = foldername

def saveSelectedFolder():
    app = QApplication(sys.argv)
    target_folder = GetFolderToLoad()
    return target_folder.foldername

def saveSelectedFile():
    app = QApplication(sys.argv)
    target_file = GetFileToSave()
    return target_file.filename

def loadSelectedFile():
    app = QApplication(sys.argv)
    target_file = GetFilesToLoad()
    return target_file.filename

def writeToDATA(file_path, data, write_mode='w'):
    with open(file_path, write_mode) as f:
        f.write(json.dumps(data))
        f.write('\n')

def setOutputPath(video_name, output_name):
    file_dir = repo_path + data_dir + video_name + '/'
    print(file_dir)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    
    output_path = file_dir + output_name + ".data"
    video_out_path = file_dir + output_name + ".mp4"
    return video_out_path, output_path

def getVideoPath(video_name):
    video_name_ext = video_name + ".mp4"
    if(video_name_ext == "None"):
        print("No video found")
        return
    video_path = repo_path + videos_dir + video_name_ext
    return video_path

def setMetadata(video_name, mapping, pairs, summary="None"):
    if video_name==0:
        video_path = 0
    
        cap = cv2.VideoCapture(video_path)

        length = 0
        fps = 0
    else:
        video_path = getVideoPath(video_name)
    
        print(video_path)
        cap = cv2.VideoCapture(video_path)

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

    _, image = cap.read()
    frame_width = image.shape[1]
    frame_height = image.shape[0]

    cap.set(2,0.0)

    angles_names = ["Knee <- Ankle -> Ground", "Hip <- Knee -> Ankle", "Shoulder <- Hip -> Knee", "Elbow <- Shoulder -> Hip", "Wrist <- Elbow -> Shoulder"]
    
    file_metadata = {
        'video_name': video_name,
        'n_frames': length,
        'n_points': len(mapping),
        'frame_width': frame_width,
        'frame_height': frame_height,
        'fps': fps,
        'keypoints_names': mapping,
        'keypoints_pairs': pairs,
        "angles_names" : angles_names,
        'summary': summary
    }
    return file_metadata

def getPoseParameters(nn_model, pose_model):
    if nn_model == "BODY_25":
        nn_mapping = keypoints_mapping_BODY_25
    else:
        nn_mapping = keypoints_mapping_BODY_25

    if pose_model == "SL":
        mapping = SL_mapping
        pairs = SL_pairs
    elif pose_model == "SR":
        mapping = SR_mapping
        pairs = SR_pairs
    elif pose_model == "Tennis":
        mapping = Tennis_mapping
        pairs = Tennis_pairs
    else:
        mapping = keypoints_mapping_BODY_25
        pairs = BODY_25_pairs

    return nn_mapping, mapping, pairs

def get_curr_screen_geometry():
    """
    Workaround to get the size of the current screen in a multi-screen setup.

    Returns:
        geometry (str): The standard Tk geometry string.
            [width]x[height]+[left]+[top]
    """
    root = tk.Tk()
    root.update_idletasks()
    root.attributes('-fullscreen', True)
    root.state('iconic')
    geometry = root.winfo_geometry()
    root.destroy()
    geometry = geometry.split('x')
    width = geometry[0]
    height = (geometry[1].split('+'))[0]
    return int(width), int(height)
def getPixel(coord_xy, f_height, mmppx=1, mmppy=1):
    j = int(coord_xy[0]/mmppx)
    i = int(f_height - (coord_xy[1]/mmppy))
    return (j, i)

def getCoord(pixel_ji, f_height, mmppx=1, mmppy=1):
    x = pixel_ji[0]*mmppx
    y = (f_height - pixel_ji[1])*mmppy
    return (x, y)

def changeKeypointsVector(personwise_keypoints, keypoints_list):
    unsorted_keypoints = -1*np.ones([len(personwise_keypoints), n_points, 2])
    for n in range(len(personwise_keypoints)):
        for i in range(n_points):
            index = personwise_keypoints[n][i]
            if index == -1:
                continue
            unsorted_keypoints[n][i] = keypoints_list[int(personwise_keypoints[n][i])][0:2]
    return unsorted_keypoints

def readFrameDATA(file_path, frame_n=0):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i==0:
                metadata = json.loads(line)
            elif i==frame_n+1:
                data = json.loads(line)
    keypoints = np.array(data["keypoints"]).astype(float)
    return metadata, keypoints

def readMultipleFramesDATA(file_path, frames=[0]):
    keypoints_vector = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if (i+1) in frames:
                data = json.loads(line)
                keypoints_vector.append(data["keypoints"])
    keypoints_vector = np.array(keypoints_vector).astype(float)
    return keypoints_vector

def readFrameJSON(file_path, frame_n=0):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i==0:
                metadata = json.loads(line)
            elif i==frame_n+1:
                data = json.loads(line)
    return metadata, data

def parse_pickle_file(file_path):
    data = {}
    # Load data
    with open(file_path, 'rb') as f:
        try:
            while True:
                data.update({pickle.load(f): pickle.load(f)})
    
        except EOFError:
            pass
    
    var_names = []
    for k, v in data.items():
        var_names.append(k)

    return data, var_names

def parse_data_file(file_path):
    keypoints_vec = []
    angles_vec = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i==0:
                metadata = json.loads(line)
            else:
                data_line = json.loads(line)
                keypoints_vec.append(data_line["keypoints"])
                try:
                    angles_vec.append(data_line["angles"])
                except:
                    angles = []
    data = {}
    data["keypoints"] = np.array(keypoints_vec).astype(float)
    data["angles"]  = np.array(angles_vec).astype(float)
    data["metadata"] = metadata
    var_names = ["metadata", "keypoints", "angles"]
    return data, var_names

def readAllFramesDATA(file_path):
    keypoints_vec = []
    angles_vec = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i==0:
                metadata = json.loads(line)
            else:
                data = json.loads(line)
                keypoints_vec.append(data["keypoints"])
                try:
                    angles_vec.append(data["angles"])
                except:
                    angles = []
    keypoints_vec = np.array(keypoints_vec).astype(float)
    angles_vec = np.array(angles_vec).astype(float)
    return metadata, keypoints_vec, angles_vec

def getFrame(video_name, n, input_path=False, allvid=False):
    if input_path:
        input_source = video_name
    else:
        if allvid:
            input_source = allvid_dir + video_name
        else:
            input_source = videos_dir + video_name
    
    cap = cv2.VideoCapture(input_source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    has_frame, image = cap.read()
    cap.release() 
    frame_width = image.shape[1]
    frame_height = image.shape[0]
    
    return image, frame_width, frame_height

def getFrames(video_name, n):
    input_source = videos_dir + video_name
    cap = cv2.VideoCapture(input_source)
    frames = []
    for i in n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        has_frame, image = cap.read()
        frames.append(image)
    
    cap.release()
    
    return frames


def angle3pt(a, b, c):
    if b in (a, c):
        raise ValueError("Undefined angle, two identical points", (a, b, c))
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    if ang < 0:
        ang += 360
    if ang > 180:
        ang = 360 - ang
    return ang

def rectangularArea(person):
    try:
        # print(person)
        max_x = max(person[:, 1])
        min_x = min([n for n in person[:, 1] if n>0])
        max_y = max(person[:, 0])
        min_y = min([n for n in person[:, 0] if n>0])
    except:
        #print(person)
        return 0
    return (max_x - min_x)*(max_y - min_y)