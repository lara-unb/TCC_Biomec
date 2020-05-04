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
from support import readAllFramesDATA, saveFileDialog, readFileDialog, saveDATAtoFile
from preprocessing import organizeBiggestPerson, selectJoints, fillwInterp
from visualizations import poseDATAtoFrame, rectAreatoFrame, showFrame
from kinematics import getRowingAngles
sys.path.append("../postprocessing")
from kalman_processing import processing_function

def keypointsToVideo(video_path, file_metadata, keypoints_vec, video_out_path=None, 
                    save_video=True, show_video=True, show_frame=False):

    frame_width, frame_height, fps = file_metadata["frame_width"], file_metadata["frame_height"], file_metadata["fps"]
    keypoints_names, keypoints_pairs = file_metadata["keypoints_names"], file_metadata["keypoints_pairs"]
    cap = cv2.VideoCapture(video_path)

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        vid_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (frame_width,frame_height))

    if(cap.isOpened() == False):
        print("Error opening video stream or file")

    for i in range(len(keypoints_vec)):
        # Process Image
        ret, imageToProcess = cap.read()
        if not ret:
            break

        # Start timer
        timer = cv2.getTickCount()

        pose_keypoints = keypoints_vec[i]


        img_out = poseDATAtoFrame(imageToProcess, pose_keypoints, 0, keypoints_names, keypoints_pairs, 
                                    thickness=3, color = -1)

        if save_video:
            vid_writer.write(img_out)

        if show_video or show_frame:
            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            # Display FPS on frame
            cv2.putText(img_out, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
            
            # Display Image
            cv2.namedWindow('OpenPose', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('OpenPose', (frame_width, frame_height))
            cv2.imshow("OpenPose", img_out)

        if show_frame:
            while True:
                if(cv2.waitKey(25) & 0xFF == ord('q')):
                    break
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print('[' + str(i) + '/' + str(len(keypoints_vec)) + ']')

    cap.release()
    if save_video:
        vid_writer.release()
    cv2.destroyAllWindows()

def test_processing(data_path, video_path, data_out_path, video_out_path, interp=False, process=False, 
                    save_data=True, save_video=True, show_video=True, show_frame=False):
    # Read data file and get all frames
    file_metadata, keypoints_vec, angles_vec = readAllFramesDATA(data_path)
    # If file doesn't include angles, add them
    if len(angles_vec) == 0:
        angles_vec = getRowingAngles(keypoints_vec)
    
    # Apply interpolation
    if interp:
        keypoints_vec = fillwInterp(keypoints_vec)
    # Apply processing function
    if process:
        print("Entered")
        keypoints_vec = processing_function(keypoints_vec)
    
    if save_data:
        saveDATAtoFile(data_out_path, file_metadata, keypoints_vec, angles_vec)
    
    keypointsToVideo(video_path, file_metadata, keypoints_vec, video_out_path, 
                    save_video, show_video, show_frame)

if __name__ == "__main__":
    data_path = readFileDialog(title="Open data file")
    video_path = readFileDialog(title="Open video file")
    output_path = saveFileDialog("Save as (no extension)")
    data_out_path = output_path + ".data"
    video_out_path = output_path + ".mp4"
    test_processing(data_path, video_path, data_out_path, video_out_path, interp=True, process=True, 
                    save_data=True, save_video=True, show_video=True, show_frame=False)
