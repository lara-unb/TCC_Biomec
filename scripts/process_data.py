import os
import sys
import cv2
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from sys import platform
import argparse
sys.path.append("../src")
from support import parse_data_file, saveFileDialog, readFileDialog, saveDATAtoFile
from preprocessing import fillwInterp
from visualizations import keypointsToVideo
from kinematics import getRowingAngles
sys.path.append("../postprocessing")
from kalman_processing import processing_function

def test_processing(data_path, video_path, data_out_path, video_out_path, interp=False, process=False, 
                    save_data=True, save_video=True, show_video=True, show_frame=False):
    # Read data file and get all frames
    data, var_names = parse_data_file(data_path)
    
    # If file doesn't include angles, add them
    if len(data["angles"]) == 0:
        data["angles"] = getRowingAngles(data["keypoints"])
    # Apply interpolation
    if interp:
        data["keypoints"] = fillwInterp(data["keypoints"])
    # Apply processing function
    if process:
        data = processing_function(data)

    if save_data:
        saveDATAtoFile(data_out_path, data)
    
    keypointsToVideo(video_path, data["metadata"], data["keypoints"], video_out_path, 
                    save_video, show_video, show_frame)

if __name__ == "__main__":
    data_path = readFileDialog(title="Open data file")
    video_path = readFileDialog(title="Open video file")
    output_path = saveFileDialog("Save as (no extension)")
    data_out_path = output_path + ".data"
    video_out_path = output_path + ".mp4"
    test_processing(data_path, video_path, data_out_path, video_out_path, interp=True, process=True, 
                    save_data=True, save_video=True, show_video=True, show_frame=False)
