import numpy as np

def invert_signal(data_vec, data_time):
    data_out = (-1)*(data_vec - np.mean(data_vec)) + np.mean(data_vec) + 1
    return [data_out, data_time]