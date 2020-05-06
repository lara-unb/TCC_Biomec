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

- ([Numpy 1.18.4](https://pypi.org/project/numpy/)) (pip install numpy)
- ([Matplotlib 3.2.1](https://pypi.org/project/matplotlib/)) (pip install matplotlib)
- ([Opencv 4.2.0.34](https://pypi.org/project/opencv-python)) (pip install opencv-python)
- ([Pykalman 0.9.5](https://pypi.org/project/pykalman/))(pip install pykalman)
- ([PyQt5 5.14.2](https://pypi.org/project/PyQt5/)) (pip install PyQt5)
- Openpose (see [github](https://github.com/CMU-Perceptual-Computing-Lab/openpose))
- ([Scipy 1.4.1](https://pypi.org/project/scipy/)) (pip install opencv-python)

# Testing

- A file structure named .DATA was created to store and manipulate keypoints and angles extracted, it based on JSON and therefor readable.
- To get a .DATA file from a video using the Openpose Python API, run the script pyopenpose_save_data.py and select the video of interest. In the file, define the saggittal plane of the rowing position (SL: Saggittal Left, SR: Saggittal Rigth).


# Test post-processing by openpose.
Teste pós processamento pela openpose. 

- Primeiro passo é clonar essa pasta do git com os seguintes comandos. 
- First step is to clone this folder fron git with the following commands. 

	git clone https://github.com/lara-unb/ema_motion_analysis.git

- Para rodar o código é necessário ter todas as dependencias instaladas, para isso deve-se usar os comandos listados em 'Dependencies' ou seguir o indicado nos links correspondentes.
- To run the code it is necessary to have all the dependencies installed, for that you must use the commands listed in 'Dependencies' or follow the one indicated in the corresponding links.

- Para processar os dados, execute dentro da pasta script o código process_data (esse passo pode ser feito da forma que ficar mais confortável, a intenção é apenas executar o programa).  
- To process the data, access the script folder and run the process_data code (this step can be done in the way that is most comfortable, the intention is just to run the program).

- Ao executar esse código, será solicitado o arquivo de dados (.data) e você deve indicar o caminho da pasta exemplos e selecionar o arquivo rowing.data, em seguida será solicitado o arquivo de vídeo também localizado na pasta exemplo e com o nome rowing. Por ultimo será solicitado a pasta onde o arquivo processado será salvo e também o nome do arquivo gerado.
- When executing this code, you will be asked for the data file (.data) and you must indicate the path of the examples folder and select the file rowing.data, then you will be asked for the video file also located in the example folder and with the name rowing . Finally, you will be asked for the folder where the processed file will be saved and also the name of the generated file.

- Ao finalizar a execução, será gerados dois arquivos, um de vídeo e um de dados na pasta selecionada.
- At the end of the execution, two files will be generated, one of video and one of data in the selected folder.


-The default processing is an interpolation followed by a kalman filter, applied to each joint position coordinate axis individually.


