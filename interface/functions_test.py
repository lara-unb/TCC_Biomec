import importlib
import sys
import numpy as np
sys.path.append("../src")
from support import readFileDialog

data_vec = np.arange(9)
data_time = np.arange(9)

file_path = readFileDialog("Open function", ".py")

processing = __import__(file_path)
importlib.reload(sys.modules[file_path])
processing = __import__(file_path)
processing_function = getattr(processing, 'processing_function')