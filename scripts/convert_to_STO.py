import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("../src")
from support import readFileDialog, parse_data_file, getFrame, saveFileDialog
from kinematics import showRowingChainAnglesPlot, inverseKinematicsRowing

joint_angles = ["Knee", "Hip"]

file_path_op = readFileDialog("Open Data file (OP)", "data")
print(file_path_op)

output_path = saveFileDialog("Save without extension") + '.sto'
if output_path == '.sto':
    print("Invalid output path")
    sys.exit(-1)
else:
    print(output_path)
    output_name = output_path.split('.sto')[0].split('/')[-1]
    print(output_name)

data, var_names = parse_data_file(file_path_op)
kp_openpose = data["keypoints"]
md_openpose = data["metadata"]
ang_openpose = data["angles"]
fps_openpose = md_openpose["fps"]
angles_names = md_openpose['angles_names']
t_openpose = np.linspace(0, len(ang_openpose) * (1/fps_openpose), len(ang_openpose))

ang_openpose_anatomical = np.copy(ang_openpose)
ang_openpose_anatomical[:, 0] = 40 - ang_openpose_anatomical[:, 0]
ang_openpose_anatomical[:, 1] = ang_openpose_anatomical[:, 1] - 180
ang_openpose_anatomical[:, 2] = 180 - ang_openpose_anatomical[:, 2]

values_dic =  {"time": t_openpose}
for joint in joint_angles:
    if joint == "Ankle":
        values_dic["ankle_angle_r"] = ang_openpose_anatomical[:, 0]
    elif joint == "Knee":
        values_dic["knee_angle_r"] = ang_openpose_anatomical[:, 1]
    elif joint == "Hip":
        values_dic["hip_flexion_r"] = ang_openpose_anatomical[:, 2]

columns_line = ''.join([col + '\t' for col in values_dic.keys()])

header = "{}\nversion=1\nnRows={}\nnColumns={}\ninDegrees=yes\n\nendheader".format(output_name,
                                                                                len(ang_openpose_anatomical),
                                                                                len(values_dic.keys()))

with open(output_path, 'w+') as f:

    f.write(header)
    f.write('\n')
    f.write(columns_line)
    f.write('\n')

    for n in range(len(ang_openpose_anatomical)):
        values_list = []
        for col_name in values_dic.keys():
            values_list.append(str(values_dic[col_name][n]))
        values_line = ''.join([value + '\t' for value in values_list])
        f.write(values_line)
        f.write('\n')
