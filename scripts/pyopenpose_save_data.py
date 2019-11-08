import sys
sys.path.append('../src')
from processing import processKeypointsData

if __name__ == "__main__":
    processKeypointsData("Target_Rowing", "Target_Rowing", process_params="BIK", pose_model="SL")
    sys.exit(-1)
