import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from support import *
from detection import *
from preprocessing import *
from parameters import *

colors_t = [(0,255,0), (0,0,255), (0,0,0), (255,255,255)]
colors_2 = [[0,255,0], [255,0,0], [0,0,255], [0,255,255],[255,255,0], 
         [255,0,255], [0,255,0], [255,200,100], [200,255,100],
         [100,255,200], [255,100,200], [100,200,255], [200,100,255],
         [200,200,0], [200,0,200],[0,200,200], [0,0,0]]

data_dir = "/home/victormacedo10/0.TCC/TCC_Biomec/Data/"
videos_dir = "/home/victormacedo10/0.TCC/TCC_Biomec/Videos/"


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
        print('[' + str(i) + '/' + str(len(keypoints_vec)-1) + ']', end='\r')

    cap.release()
    if save_video:
        vid_writer.release()
    cv2.destroyAllWindows()

def visualizeColoredVideo(video_name, file_name, thickness=3, joint_names = [-1]):
    if(video_name == "None"):
        print("No video found")
        return
    if(file_name == "None"):
        print("No DATA found")
        return
    video_name = (video_name).split(sep='.')[0]
    file_dir = data_dir + video_name + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = file_dir + file_name
    metadata, keypoints_vector = readAllFramesDATA(file_path)

    video_name_ext = [filename for filename in os.listdir(videos_dir) if filename.startswith(metadata["video_name"])]
    
    joint_pairs = metadata["joint_pairs"]
    pairs = []
    for j in joint_pairs:
        pairs.append(pose_pairs[j])
    joints = np.unique(pairs)

    image, _, _ = getFrame(video_name_ext[0], 0)
    frame = np.zeros(image.shape)
    joint_name = "Neck"
    j_list = []
    if(joint_names[0] == -1):
        for i in range(len(joints)):
            j_list.append(i)
    else:
        for joint_name in joint_names:
            j_list.append(joints.tolist().index(keypoints_mapping.index(joint_name)))

    for i in range(1, len(keypoints_vector)):
        for j in range(keypoints_vector.shape[1]):
            if(j not in j_list):
                continue    
            A = tuple(keypoints_vector[i-1,j,:].astype(int))
            B = tuple(keypoints_vector[i,j,:].astype(int))
            if (-1 in A) or (-1 in B):
                continue
            if (0 in A) or (0 in B):
                continue
            cv2.line(frame, (A[0], A[1]), (B[0], B[1]), indep_colors[j], thickness, cv2.LINE_AA)
            if(i==1):
                print(indep_colors[j])
    
    showFrame(frame)
    return frame

def keypointsDATAtoFrame(image, keypoints, joint_names, joints_pairs, t_circle=3, t_line=1, color_circle = [255,0,0], color_line=[0,0,0]):
    frame_out = np.copy(image)
    for (joint_A, joint_B) in joints_pairs:
        idx_A = joint_names.index(joint_A)
        kp_A = keypoints[idx_A].astype(int)
        idx_B = joint_names.index(joint_B)
        kp_B = keypoints[idx_B].astype(int)
        if ((-1 in kp_A) or (0 in kp_A)) and ((-1 in kp_B) or (0 in kp_B)): # no A and no B
            continue
        elif ((-1 in kp_A) or (0 in kp_A)) and (not((-1 in kp_B) or (0 in kp_B))): # mo A and B
            cv2.circle(frame_out, (kp_B[0], kp_B[1]), t_circle, color_circle, -1)
            continue
        elif (not((-1 in kp_A) or (0 in kp_A))) and ((-1 in kp_B) or (0 in kp_B)): # A and no B
            cv2.circle(frame_out, (kp_A[0], kp_A[1]), t_circle, color_circle, -1)
            continue
        else: # A and B
            cv2.line(frame_out, (kp_A[0], kp_A[1]), (kp_B[0], kp_B[1]), color_line, t_line, cv2.LINE_AA)
            cv2.circle(frame_out, (kp_A[0], kp_A[1]), t_circle, color_circle, -1)
            cv2.circle(frame_out, (kp_B[0], kp_B[1]), t_circle, color_circle, -1)
    return frame_out

def visualizePoints(keypoints_list, personwise_keypoints, person, joint_n):
    index = int(personwise_keypoints[person][joint_n])
    if(index == -1):
        return -1
    X = np.int32(keypoints_list[index, 0])
    Y = np.int32(keypoints_list[index, 1])
    return (X, Y)

def visualizeFrame(video_name, n):
    n = int(np.round(n))
    if(video_name == "None"):
        print("Choose a video")
    else:
        image, _, _ = getFrame(video_name, n)
        plt.figure(figsize=[14,10])
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")

def visualizeHeatmap(image, output, joint_n, threshold=0.1, alpha=0.6, binary=False, show_point=False):
    frame_width = image.shape[1]
    frame_height = image.shape[0]
    prob_map = output[0, joint_n, :, :]
    prob_map = cv2.resize(prob_map, (frame_width, frame_height))
    if(binary == False):
        prob_map = np.where(prob_map<threshold, 0.0, prob_map)
    else:
        prob_map = np.where(prob_map<threshold, 0.0, prob_map.max())
    
    if show_point:
        return prob_map
    
    plt.figure(figsize=[9,6])
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.imshow(prob_map, alpha=alpha, vmax=prob_map.max(), vmin=0.0)
    #plt.colorbar()
    plt.axis("off")

def visualizeKeypoints(frame, personwise_keypoints, keypoints_list, persons, joint_pairs):    
    frame_out = frame.copy()
    
    if (persons[0] == -1):
        persons = np.arange(len(personwise_keypoints))
        
    if (joint_pairs[0] == -1):
        joint_pairs = np.arange(len(pose_pairs)-2)
        
    for i in joint_pairs:
        for n in persons:
            index = personwise_keypoints[n][np.array(pose_pairs[i])]
            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            cv2.line(frame_out, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

    plt.figure(figsize=[9,6])
    plt.imshow(frame_out[:,:,[2,1,0]])
    plt.axis("off")
    
def visualizeMainKeypoints(frame, pose_keypoints, persons, joint_pairs):    
    frame_out = frame.copy()
    
    if (persons[0] == -1) or (max(persons) >= len(pose_keypoints)):
        persons = np.arange(len(pose_keypoints))
    
    try:
        joint_pairs[0]
    except IndexError:
        pass
    else:
        if (joint_pairs[0] == -1):
            joint_pairs = np.arange(len(pose_pairs)-2)
    
    
    for n in persons:
        for i in joint_pairs:
            A = tuple(pose_keypoints[n][joint_pairs[i][0]].astype(int))
            B = tuple(pose_keypoints[n][joint_pairs[i][1]].astype(int))
            if (-1 in A) or (-1 in B):
                continue
            cv2.line(frame_out, (A[0], A[1]), (B[0], B[1]), colors[i], 3, cv2.LINE_AA)

    plt.figure(figsize=[9,6])
    plt.imshow(frame_out[:,:,[2,1,0]])
    plt.axis("off")

def visualizeSingleKeypoints(frame, sorted_keypoints, joint_pairs):    
    frame_out = frame.copy()
        
    if (joint_pairs[0] == -1):
        joint_pairs = np.arange(len(pose_pairs)-2)
    
    for i in joint_pairs:
        A = tuple(sorted_keypoints[pose_pairs[i][0]].astype(int))
        B = tuple(sorted_keypoints[pose_pairs[i][1]].astype(int))
        if (-1 in A) or (-1 in B):
            continue
        cv2.line(frame_out, (A[0], A[1]), (B[0], B[1]), colors[i], 3, cv2.LINE_AA)

    plt.figure(figsize=[9,6])
    plt.imshow(frame_out[:,:,[2,1,0]])
    plt.axis("off")

def pointDATAtoFrame(frame, main_keypoints, joint_pairs, point, thickness=3, color = -1):
    pairs = []
    for j in joint_pairs:
        pairs.append(pose_pairs[j])
    joints = np.unique(pairs)

    point_n = keypoints_mapping.index(point)
    joints = joints.tolist()
    if point_n in joints:
        point_idx = joints.index(point_n)
        A = tuple(main_keypoints[point_idx].astype(int))
        if (-1 in A):
            return frame

        cv2.circle(frame, (A[0], A[1]), thickness, colors_2[color], -1)

    return frame

def keypointsFromDATA(video_name, file_name, frame_n=0):
    if(video_name == "None"):
        print("No video found")
        return
    if(file_name == "None"):
        print("No DATA found")
        return
    video_name = (video_name).split(sep='.')[0]
    file_dir = data_dir + video_name + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = file_dir + file_name
    metadata, keypoints = readFrameDATA(file_path, frame_n=frame_n)
    joint_pairs = metadata["joint_pairs"]

    video_name_ext = [filename for filename in os.listdir(videos_dir) if filename.startswith(metadata["video_name"])]
    
    image, _, _ = getFrame(video_name_ext[0], frame_n)
    frame = poseDATAtoFrame(image, keypoints, joint_pairs)
    showFrame(frame)

def keypointsFromDATACompare(video_name, file_names = ['None'], file_ref = ['None'], frame_n=0,
                            show_point=False, point='Nose', thickness=3):
    if(video_name == "None"):
        print("No video found")
        return
    if(file_names[0] == "None" and file_ref == "None"):
        image, _, _ = getFrame(video_name, frame_n)
        showFrame(image)
        return

    frame, _, _ = getFrame(video_name, frame_n)

    video_name = (video_name).split(sep='.')[0]
    file_dir = data_dir + video_name + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    
    if(file_ref != "None"):
        file_path = file_dir + file_ref
        metadata, keypoints = readFrameDATA(file_path, frame_n=frame_n)
        joint_pairs = metadata["joint_pairs"]
        if show_point:
            frame = pointDATAtoFrame(frame, keypoints, joint_pairs, point, thickness, color=0)
        else:
            frame = poseDATAtoFrame(frame, keypoints, joint_pairs, thickness, color=0)

    if(file_names[0] == "None"):
        showFrame(frame)
        return

    for i in range(len(file_names)):
        file_path = file_dir + file_names[i]
        metadata, keypoints = readFrameDATA(file_path, frame_n=frame_n)
        joint_pairs = metadata["joint_pairs"]
        if show_point:
            frame = pointDATAtoFrame(frame, keypoints, joint_pairs, point, thickness, color=i+1)
        else:
            frame = poseDATAtoFrame(frame, keypoints, joint_pairs, thickness, color=i+1)
    
    showFrame(frame)
    
def keypointsFromJSON(video_name, file_name, persons, custom, joint_pairs, frame_n=0,
                      threshold=0.1, n_interp_samples=10, paf_score_th=0.1, conf_th=0.7, 
                      read_file = False):
    if(video_name == "None"):
        print("No video found")
        return
    if(file_name == "None"):
        print("No JSON found")
        return
    video_name = (video_name).split(sep='.')[0]
    file_dir = data_dir + video_name + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = file_dir + file_name
    metadata, data = readFrameJSON(file_path, frame_n=frame_n)

    if read_file:
        try:
            metadata["threshold"]
        except KeyError:
            pass
        else:
            threshold = metadata["threshold"]
            n_interp_samples = metadata["n_interp_samples"]
            paf_score_th = metadata["paf_score_th"]
            conf_th = metadata["conf_th"]

    output = np.array(data["output"]).astype(float)
    frame_height, frame_width = metadata["frame_height"], metadata["frame_width"]
    personwise_keypoints, keypoints_list = keypointsFromHeatmap(output, frame_width, frame_height, threshold, 
                     n_interp_samples, paf_score_th, conf_th)
    
    
    video_name_ext = [filename for filename in os.listdir(videos_dir) if filename.startswith(metadata["video_name"])]
    image, _, _ = getFrame(video_name_ext[0], frame_n)
  
    if persons == 'Biggest':
        p = [custom]
        sorted_keypoints = organizeBiggestPerson(personwise_keypoints, keypoints_list)
        visualizeMainKeypoints(image, sorted_keypoints, p, joint_pairs)
    elif persons == 'Unsorted':
        p = [custom]
        unsorted_keypoints = changeKeypointsVector(personwise_keypoints, keypoints_list)
        visualizeMainKeypoints(image, unsorted_keypoints, p, joint_pairs)
    else:
        p = [-1]
        unsorted_keypoints = changeKeypointsVector(personwise_keypoints, keypoints_list)
        visualizeMainKeypoints(image, unsorted_keypoints, p, joint_pairs)
        
def heatmapFromJSON(video_name, file_name, joint_n, threshold, alpha, binary, 
                     n_interp_samples, paf_score_th, conf_th, frame_n=0, show_point=False):
    if(video_name == "None"):
        print("No video found")
        return
    if(file_name == "None"):
        print("No JSON found")
        return
    video_name = (video_name).split(sep='.')[0]
    file_dir = data_dir + video_name + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = file_dir + file_name
    metadata, data = readFrameJSON(file_path, frame_n=frame_n)
    output = np.array(data["output"]).astype(float)
    frame_height, frame_width = metadata["frame_height"], metadata["frame_width"]
    personwise_keypoints, keypoints_list = keypointsFromHeatmap(output, frame_width, frame_height, threshold, 
                     n_interp_samples, paf_score_th, conf_th)
    
    video_name_ext = [filename for filename in os.listdir(videos_dir) if filename.startswith(metadata["video_name"])]
    image, _, _ = getFrame(video_name_ext[0], frame_n)    
    
    prob_map = visualizeHeatmap(image, output, joint_n, threshold, alpha, binary, show_point)
    
    if show_point:  
        fig = plt.figure(figsize=[9,6])
        ax = plt.gca()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i in range(len(personwise_keypoints)):
            A = visualizePoints(keypoints_list, personwise_keypoints, i, joint_n)
            if(A != -1):
                cv2.circle(image, A, 3, [255,255,255], -1, cv2.LINE_AA)
        plt.imshow(image)
        plt.imshow(prob_map, alpha=alpha, vmax=prob_map.max(), vmin=0.0)
        #plt.colorbar()
        plt.axis("off")
        
def showFrame(frame, figsize=[9,6], savefig=False, figname=None):
    if figsize == None:
        plt.figure()
    else:
        plt.figure(figsize=figsize)
    plt.imshow(frame[:,:,[2,1,0]])
    plt.axis("off")
    if savefig:
        plt.savefig("../images/" + figname + ".png", bbox_inches="tight", pad_inches = 0)
    plt.show()

def rectAreatoFrame(frame, pose_keypoints):
    try:
        print(pose_keypoints.shape[0])
        for n in range(pose_keypoints.shape[0]):
            area = rectangularArea(pose_keypoints[n,:,:2])
            max_x , min_x, max_y, min_y = getVertices(pose_keypoints[n,:,:2])
            print(max_x , min_x, max_y, min_y)
            cv2.rectangle(frame, (min_y, min_x), (max_y, max_x), thickness = 2)
            print("{0}: {1}".format(n, area))
        return frame
    except IndexError:
        return frame

def poseDATAtoFrame(frame, pose_keypoints, persons, joint_names, pairs_names, thickness=3, color = -1):
    try:
        single = False
        if persons == 0:
            persons = [0]
            single = True
        if persons == -1:
            persons = np.arange(pose_keypoints.shape[0])

    except IndexError:
        return frame

    try:
        for n in persons:
            i = 0
            for pair in pairs_names:
                A_idx = joint_names.index(pair[0])
                B_idx = joint_names.index(pair[1])
                if single:
                    A = tuple(pose_keypoints[A_idx][:2].astype(int))
                    B = tuple(pose_keypoints[B_idx][:2].astype(int))
                else:
                    A = tuple(pose_keypoints[n][A_idx][:2].astype(int))
                    B = tuple(pose_keypoints[n][B_idx][:2].astype(int))
                if (0 in A) or (0 in B):
                    i+=1
                    continue
                if(color == -1):
                    cv2.line(frame, (A[0], A[1]), (B[0], B[1]), colors_25[i], thickness, cv2.LINE_AA)
                else:
                    cv2.line(frame, (A[0], A[1]), (B[0], B[1]), colors_2[color], thickness, cv2.LINE_AA)
                i+=1
    except IndexError:
        return frame
    return frame

def poseDATAtoCI(frame, keypoints_vector, thickness=3):
    frame = np.zeros(frame.shape)

    for i in range(1, len(keypoints_vector)):
        for j in range(keypoints_vector.shape[1]): 
            A = tuple(keypoints_vector[i-1,j,:].astype(int))
            B = tuple(keypoints_vector[i,j,:].astype(int))
            if (-1 in A) or (-1 in B):
                continue
            if (0 in A) or (0 in B):
                continue
            cv2.line(frame, (A[0], A[1]), (B[0], B[1]), indep_colors[j], thickness, cv2.LINE_AA)
            if(i==1):
                print(indep_colors[j])
    return frame