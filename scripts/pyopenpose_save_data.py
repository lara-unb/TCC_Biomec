import sys
sys.path.append('../src')
from processing import processKeypointsData
from support import loadSelectedFile

if __name__ == "__main__":
    file_path = loadSelectedFile()
    processKeypointsData(file_path, process_params="BIK", pose_model="SL")
    sys.exit(-1)
