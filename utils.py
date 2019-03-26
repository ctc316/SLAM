from bisect import bisect_left

import numpy as np
import matplotlib.pyplot as plt




'''
Δs = (Δr + Δl) / 2
Δθ = (Δr - Δl) / wheel_distance
Δx = Δs cos(θ + Δθ/2)
Δy = Δs sin(θ + Δθ/2)
'''
def getOdometry(data, sample_rate, offset=0):
    wheel_diameter = (584.2 - 330.2) / 1000.0 # meter
    wheel_distance = (311.15 + 476.25) / 2 / 1000.0 # meter
    distance = np.pi * wheel_diameter / 360  # 360 per revolution
    scale = 1.85
    
    right = data[0] * distance
    left  = data[1] * distance
    theta = (right - left) / wheel_distance / scale
    theta = np.add.accumulate(theta)
    x = ((right + left) / 2) * np.cos(theta)
    y = ((right + left) / 2) * np.sin(theta)
    xx = np.add.accumulate(x, axis=0)
    yy = np.add.accumulate(y, axis=0)
    
    return xx[offset::sample_rate], yy[offset::sample_rate], theta[offset::sample_rate], data[4][offset::sample_rate]


def getMatchedImuData(data, timestamps):
    ts_list = data[6]
    res = []
    for t in timestamps:
        idx = bisect_left(ts_list, t)
        if idx == 0:
            pass
        elif idx == len(ts_list):
            idx = len(ts_list) - 1
        elif abs(t - ts_list[idx - 1]) < abs(t - ts_list[idx]):
            idx -= 1        
        res.append(np.array([d[idx] for d in data]))
    return np.array(res)

def isTilted(gyro_y):
    return abs(gyro_y) > 0.03


def getMatchedLidarData(data, timestamps):
    ts_list = [d['t'] for d in data]
    res = []
    for t in timestamps:
        idx = bisect_left(ts_list, t)
        if idx == 0:
            pass
        elif idx == len(ts_list):
            idx = len(ts_list) - 1
        elif abs(t - ts_list[idx - 1]) < abs(t - ts_list[idx]):
            idx -= 1        
        res.append(data[idx])
    return np.array(res)    


def getStaticOffset(data):
    offset = 0
    for i in range(1, len(data[0])):
        if data[0][i] - data[0][i - 1] != 0:
            offset = i
            break
            
    return offset