# Automatic Rowing Biomechanics Analysis

Project (under development) using Openpose as a markerless pose estimation tool along with addition processing, in order to acquire biomechanical parameters such as stroke cadence and body angulations. 

This work is situated in a larger project, (see [Project EMA website](http://projectema.com)), where motion analysis is used in order to evalute control parameters for electroestimulation movement in spinal cord injury athletes.

The project consists of mainly three steps, presented in the diagram below:

<p align="center">
  <img src="https://github.com/lara-unb/ema_motion_analysis/blob/master/images/Visão_geral.png?raw=true" alt="Project's block diagram"/>
</p>

The result of the processing is depicted in the following image, indicating the main joints of interest in the sagittal plane. 

<p align="center">
  <img src="https://github.com/lara-unb/ema_motion_analysis/blob/master/images/kp.png?raw=true" alt="Rowing pose estimation"/>
</p>

Then, using those joint coordinates acquired, angles are deduced: 
 
<p align="center">
  <img src="https://github.com/lara-unb/ema_motion_analysis/blob/master/images/ang_1.png?raw=true" alt="Rowing angles calculation"/>
</p>

# Dependencies

- Numpy (pip install numpy)
- Matplotlib (pip install matplotlib)
- Opencv (pip install opencv-python)
- Pykalman (pip install pykalman)
- PyQt5 (pip install PyQt5)
- Openpose (see [github](https://github.com/CMU-Perceptual-Computing-Lab/openpose))

# Testing

- A file structure named .DATA was created to store and manipulate keypoints and angles extracted, it based on JSON and therefor readable.
- To get a .DATA file from a video using the Openpose Python API, run the script pyopenpose_save_data.py and select the video of interest. In the file, define the saggittal plane of the rowing position (SL: Saggittal Left, SR: Saggittal Rigth).
- To process the data run the script process_data.py and select the .DATA file, the video used to generate the .DATA file, and their respective output destinations. The default processing is an interpolation followed by a kalman filter, applied to each joint position coordinate axis individually.
- A video and .DATA example is provided under the examples folder.