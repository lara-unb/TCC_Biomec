import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import pprint
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.signal import find_peaks
import ipywidgets as wg
from IPython.display import display, HTML
import sys

def defineRowingPhases(video_path, n_frames, frame_no=0):
    cap = cv2.VideoCapture(video_path)
    pose_frame = [0,0,0,0,0]
    moment = 0
    if(cap.isOpened() == False):
        print("Error opening video stream or file")
        print(video_path)
        sys.exit(-1)
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret_val, img = cap.read()
        if ret_val:
            img = cv2.resize(img, (960, 540)) 
            cv2.imshow('Video Analysis', img)
        arg = cv2.waitKey(1)
        if arg == 27:
            break  # esc to quit
        elif arg == 13: # pressed enter
            pose_frame[moment] = frame_no
            moment = (moment+1)%5
            print(pose_frame)
        elif arg == 100:
            frame_no = (frame_no+1)%n_frames
        elif arg == 97:
            frame_no = (frame_no-1)%n_frames
    cap.release()
    cv2.destroyAllWindows()
    return pose_frame

def plotRowingMoments(angles_vec, pose_frame, fps, angles_names, savefig=False, figname=None):
    markers=['^', 's', 'p', 'h', '8']
    rowing_phases = ['Catch', 'Leg Drive', 'Arm Drive', 'Arm Recovery', 'Leg Recovery']
    phases_colors = ['black', 'gray', 'red', 'green', 'purple']
    n0 = pose_frame[0]
    n1 = pose_frame[4]
    angle_knee = "Hip <- Knee -> Ankle"
    angle_elb = "Wrist <- Elbow -> Shoulder"
    angles_vec_knee = angles_vec[n0:n1+1, angles_names.index(angle_knee)]
    angles_vec_elb = angles_vec[n0:n1+1, angles_names.index(angle_elb)]
    time_vec_cycle = np.linspace(n0*(1/fps), n1*(1/fps), n1+1-n0)
    plt.figure()
    plt.plot(time_vec_cycle, angles_vec_knee, label='Knee')
    plt.plot(time_vec_cycle, angles_vec_elb, label='Elbow', linestyle='-.')
    for i in range(len(rowing_phases)):
        plt.scatter(time_vec_cycle[pose_frame[i]-pose_frame[0]], angles_vec_knee[pose_frame[i]-pose_frame[0]], 
                marker=markers[i], color=phases_colors[i], label=rowing_phases[i])
        plt.scatter(time_vec_cycle[pose_frame[i]-pose_frame[0]], angles_vec_elb[pose_frame[i]-pose_frame[0]], 
                marker=markers[i], color=phases_colors[i])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel("Angle (degrees)")
    plt.xlabel("Time (s)")
    plt.grid(True)
    if savefig:
        plt.savefig("../images/" + figname + ".png", bbox_inches='tight')
    plt.show()

def compareRowingMoments(angles_vec_ref, pose_frame_ref, fps_ref, 
                         angles_vec_target, pose_frame_target, fps_target, 
                         angles_names):
    rowing_phases = ['Catch', 'Leg Drive', 'Arm Drive', 'Arm Recovery', 'Leg Recovery']
    phases_colors = ['black', 'gray', 'red', 'green', 'purple']
    angle_knee = "Hip <- Knee -> Ankle"
    angle_elb = "Wrist <- Elbow -> Shoulder"
    n0_ref, n0_target = pose_frame_ref[0], pose_frame_target[0]
    n1_ref, n1_target = pose_frame_ref[4], pose_frame_target[4]
    angles_vec_knee_ref = angles_vec_ref[n0_ref:n1_ref+1, angles_names.index(angle_knee)]
    angles_vec_elb_ref = angles_vec_ref[n0_ref:n1_ref+1, angles_names.index(angle_elb)]
    angles_vec_knee_target = angles_vec_target[n0_target:n1_target+1, angles_names.index(angle_knee)]
    angles_vec_elb_target = angles_vec_target[n0_target:n1_target+1, angles_names.index(angle_elb)]
    time_vec_cycle_ref = np.linspace(n0_ref*(1/fps_ref), n1_ref*(1/fps_ref), n1_ref+1-n0_ref) - n0_ref*(1/fps_ref)
    time_vec_cycle_target = np.linspace(n0_target*(1/fps_target), n1_target*(1/fps_target), n1_target+1-n0_target) - n0_target*(1/fps_target)
    plt.figure(figsize=[7,5])
    plt.plot(time_vec_cycle_ref, angles_vec_knee_ref, label='Ref Knee')
    plt.plot(time_vec_cycle_ref, angles_vec_elb_ref, label='Ref Elbow')
    plt.plot(time_vec_cycle_target, angles_vec_knee_target, label='Target Knee')
    plt.plot(time_vec_cycle_target, angles_vec_elb_target, label='Target Elbow')
    for i in range(len(rowing_phases)):
        plt.scatter(time_vec_cycle_ref[pose_frame_ref[i]-pose_frame_ref[0]], 
                    angles_vec_knee_ref[pose_frame_ref[i]-pose_frame_ref[0]], 
                    marker='o', color=phases_colors[i], label=rowing_phases[i])
        plt.scatter(time_vec_cycle_ref[pose_frame_ref[i]-pose_frame_ref[0]], 
                    angles_vec_elb_ref[pose_frame_ref[i]-pose_frame_ref[0]], 
                    marker='o', color=phases_colors[i])
        plt.scatter(time_vec_cycle_target[pose_frame_target[i]-pose_frame_target[0]], 
                    angles_vec_knee_target[pose_frame_target[i]-pose_frame_target[0]], 
                    marker='o', color=phases_colors[i])
        plt.scatter(time_vec_cycle_target[pose_frame_target[i]-pose_frame_target[0]], 
                    angles_vec_elb_target[pose_frame_target[i]-pose_frame_target[0]], 
                    marker='o', color=phases_colors[i])
    plt.title("Rowing Angles for One Cycle")
    plt.legend()
    plt.ylabel("Angle (degrees)")
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.show()
    
def compareSingleJointPhases(row_cycle_target, time_row_cycle_target, pose_frame_target,
                             pose_frame_estimated, angle_name='Knee'):
    
    rowing_phases = ['Catch GT', 'Leg Drive GT', 'Arm Drive GT', 'Arm Recovery GT', 'Leg Recovery GT']
    rowing_phases_est = ['Catch Est', 'Leg Drive Est', 'Arm Drive Est', 'Arm Recovery Est', 'Leg Recovery Est']
    phases_colors = ['black', 'gray', 'red', 'green', 'purple']
    plt.figure(figsize=[7,5])
    plt.plot(time_row_cycle_target, row_cycle_target)
    for i in range(len(rowing_phases)):
            plt.scatter(time_row_cycle_target[pose_frame_target[i]-pose_frame_target[0]], 
                        row_cycle_target[pose_frame_target[i]-pose_frame_target[0]], 
                        marker='o', color=phases_colors[i], label=rowing_phases[i])
            plt.scatter(time_row_cycle_target[pose_frame_estimated[i]-pose_frame_estimated[0]], 
                        row_cycle_target[pose_frame_estimated[i]-pose_frame_estimated[0]], 
                        marker='^', color=phases_colors[i], label=rowing_phases_est[i])
    plt.legend()
    plt.ylabel("Angle (degrees)")
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.show()

def showAnglesRowing(frame, pose_keypoints, angles, thickness=2, textsize=2, adj=8, alpha=1.0, rect_color=(50,170,50), text_color=(0,0,0)):
    frame_ang = frame.copy()
    for i in range(len(angles)):
        str_to_write = str(int(round(angles[i])))
        font = cv2.FONT_HERSHEY_DUPLEX
        text_len = cv2.getTextSize(str_to_write, font, textsize, thickness)[0]
        A = pose_keypoints[i]
        B = pose_keypoints[i+1]
        D = (int(A[0]), int(A[1]))
        if i==4:
            D = (int(A[0]) - int(text_len[0]/2), int(A[1]))
            cv2.rectangle(frame_ang, (D[0]-adj, D[1]+adj), (D[0] + text_len[0]+adj, D[1] - text_len[1]-adj), rect_color, -1)
            cv2.putText(frame_ang, str_to_write, D, font, textsize,text_color,thickness,cv2.LINE_AA)
        elif i==1:
            D = (int(A[0]) - int(text_len[0]/2), int(A[1]) + int(text_len[1]))
            cv2.rectangle(frame_ang, (D[0]-adj, D[1]+adj), (D[0] + text_len[0]+adj, D[1] - text_len[1]-adj), rect_color, -1)
            cv2.putText(frame_ang, str_to_write, D, font, textsize,text_color,thickness,cv2.LINE_AA)
        else:
            D = (int(A[0]) - int(text_len[0]/2), int(A[1]) + int(text_len[1]/2))
            cv2.rectangle(frame_ang, (D[0]-adj, D[1]+adj), (D[0] + text_len[0]+adj, D[1] - text_len[1]-adj), rect_color, -1)
            cv2.putText(frame_ang, str_to_write, D, font, textsize,text_color,thickness,cv2.LINE_AA)
        i+=1
        # Following line overlays transparent rectangle over the image
    image_new = cv2.addWeighted(frame_ang, alpha, frame, 1 - alpha, 0)
    return image_new

def plotDTWPaths(row_cycle_ref, row_cycle_target, time_row_cycle_ref, time_row_cycle_target, paths):
    plt.figure()
    for (map_row_cycle_ref, map_row_cycle_target) in paths:
        plt.plot([time_row_cycle_target[map_row_cycle_target], time_row_cycle_ref[map_row_cycle_ref]], [row_cycle_target[map_row_cycle_target], row_cycle_ref[map_row_cycle_ref]], 'r')
    plt.plot(time_row_cycle_target, row_cycle_target, color = 'b', label = 'Target', linewidth=5)
    plt.plot(time_row_cycle_ref, row_cycle_ref, color = 'g',  label = 'Reference', linewidth=5)
    plt.legend()
    plt.ylabel("Angle (degrees)")
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.show()

def movementAnalysis():
    pass