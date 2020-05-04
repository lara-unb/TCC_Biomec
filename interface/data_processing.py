'''
This file is a collection of useful functions for dealing with rowing data, including open and saving files, IMU data,
EMG data and data syncing.
Author: Lucas Fonseca
Contact: lucasafonseca@lara.unb.br
Date: Feb 25th 2019
'''

from PyQt5.QtWidgets import QWidget, QFileDialog
from transformations import euler_from_quaternion
from pyquaternion import Quaternion
from numpy import mean
import numpy as np
import math
import sys

class IMU:

    def __init__(self, this_id):
        self.id = this_id
        self.timestamp = []
        self.x_values = []
        self.y_values = []
        self.z_values = []
        self.w_values = []
        self.euler_x = []
        self.euler_y = []
        self.euler_z = []
        self.acc_x = []
        self.acc_y = []
        self.acc_z = []

    def get_euler_angles(self):
        for i in range(len(self.timestamp)):
            # [self.euler_x[i], self.euler_y[i], self.euler_z[i]] =
            euler = euler_from_quaternion((self.x_values[i],
                                           self.y_values[i],
                                           self.z_values[i],
                                           self.w_values[i]))
            self.euler_x.append(euler[0])
            self.euler_y.append(euler[1])
            self.euler_z.append(euler[2])


def lpf(x, cutoff, fs, order=5):
    import numpy as np
    from scipy.signal import filtfilt, butter
    """
    low pass filters signal with Butterworth digital
    filter according to cutoff frequency

    filter uses Gustafsson’s method to make sure
    forward-backward filt == backward-forward filt

    Note that edge effects are expected

    Args:
        x      (array): signal data (numpy array)
        cutoff (float): cutoff frequency (Hz)
        fs       (int): sample rate (Hz)
        order    (int): order of filter (default 5)

    Returns:
        filtered (array): low pass filtered data
    """
    nyquist = fs / 2
    b, a = butter(order, cutoff / nyquist)
    if not np.all(np.abs(np.roots(a)) < 1):
        raise Exception('Filter with cutoff at {} Hz is unstable given '
                         'sample frequency {} Hz'.format(cutoff, fs))
    filtered = filtfilt(b, a, x, method='gust')
    return filtered

def median_filter(x, size):
    import numpy as np
    if size % 2 != 0:
        raise Exception('size must be odd')
    halfish = (size - 1) / 2
    out = x[0:halfish]
    for i in range(halfish + 1, len(x) - halfish):
        out.append(np.mean(np.array(x[i - halfish:i + halfish])))
    out = out + x[-halfish:]
    if not len(x) == len(out):
        raise Exception('Out vector with different size than input vector')

def filter_emg(emg_data):
    values_to_pop = []
    j = len(emg_data)
    try:
        for i in range(j):
            if emg_data[j] == -1:
                # values_to_pop.append(i)
                emg_data.pop(i)
            else:

                j = + 1
    except Exception:
        pass
    # TODO implement filter here
    norm = [i/max(emg_data) for i in emg_data]
    return mean(norm)

def run_dash(app_dash):
    app_dash.run_server(debug=True)

# Method for syncing data from sources with different sample rates, or inconsistent ones.
def resample_series(x1, y1, x2, y2, freq=100):
    from numpy import floor, ceil, arange

    if len(x1) != len(y1) or len(x2) != len(y2):
        print('Unequal lengths.')
        return -1

    period = 1/freq
    real_start_time = min(x1[0], x2[0])
    start_time = floor(real_start_time / period) * period
    real_final_time = max(x1[-1], x2[-1])
    final_time = ceil(real_final_time / period) * period

    time = arange(start_time, final_time, period)

    y1_i = 0
    y2_i = 0
    y1_out = []
    y2_out = []

    for t in time:
        y1_out.append(y1[y1_i])
        y2_out.append(y2[y2_i])

        while (t + period) > x1[y1_i] > t and y1_i < (len(x1) - 1):
            y1_i += 1
        while (t + period) > x2[y2_i] > t and y2_i < (len(x2) - 1):
            y2_i += 1

    return [time, y1_out, y2_out]

def div_filter(data, factor):
    out = []
    for i in range(0, len(data), factor):
        out.append(data[i])
    return out

def calculate_accel(acc_x, acc_y, acc_z, i):
    import numpy as np
    out = np.sqrt(np.power(acc_x[i], 2) + np.power(acc_y[i], 2) + np.power(acc_z[i], 2))
    return out

def correct_fes_input(button_timestamp, stim_state):
    wrong_descend = 0
    for i in range(1, len(stim_state)):
        if stim_state[i] == 0 and stim_state[i-1] == 1:
            wrong_descend = i
        if stim_state[i] == 1 and stim_state[i-1] == 0 and wrong_descend != 0:
            for j in range(wrong_descend, i):
                stim_state[j] = 1
            wrong_descend = 0
    return stim_state

def make_quaternions(imu):
    q = []
    for i in range(len(imu.resampled_x)):
        try:
            q.append(Quaternion(imu.resampled_w[i],
                                imu.resampled_x[i],
                                imu.resampled_y[i],
                                imu.resampled_z[i]
                                ))
        except Exception:
            pass
    return q

def angle(q):
    try:
        qr = q.elements[0]
        if qr > 1:
            qr = 1
        elif qr < -1:
            qr = -1
        angle = 2 * math.acos(qr)
        angle = angle * 180 / math.pi
        if angle > 180:
            new_angle = 360 - angle
        return angle
    except Exception as e:
        print('Exception "' + str(e) + '" in line ' + str(sys.exc_info()[2].tb_lineno))


def generate_imu_data(imus, t, imu_forearm_id, imu_arm_id):
    if imus[0].id == imu_forearm_id:
        imu_0 = 0
        imu_1 = 1
    else:
        imu_1 = 0
        imu_0 = 1

    q0 = make_quaternions(imus[imu_0])
    q1 = make_quaternions(imus[imu_1])

    q = []
    [q.append(i * j.conjugate) for i, j in zip(q0, q1)]

    qx = []
    qy = []
    qz = []
    qw = []
    qang = []
    acc_x_0 = [i for i in imus[imu_0].resampled_acc_x]
    acc_y_0 = [i for i in imus[imu_0].resampled_acc_y]
    acc_z_0 = [i for i in imus[imu_0].resampled_acc_z]
    acc_x_1 = [i for i in imus[imu_1].resampled_acc_x]
    acc_y_1 = [i for i in imus[imu_1].resampled_acc_y]
    acc_z_1 = [i for i in imus[imu_1].resampled_acc_z]

    acc_0 = [acc_x_0, acc_y_0, acc_z_0]
    acc_1 = [acc_x_1, acc_y_1, acc_z_1]

    acc = [acc_0, acc_1]

    for quat in q:
        qw.append(quat.elements[0])
        qx.append(quat.elements[1])
        qy.append(quat.elements[2])
        qz.append(quat.elements[3])
        qang.append(angle(quat))

    dqang = np.append([0], np.diff(qang) / np.diff(t))

    return qang, dqang, acc

def mean_std_features(X):
    return np.mean(X), np.std(X)

def scale_features(X, X_mean, X_std, single=True):
    return ((X - X_mean)/X_std)