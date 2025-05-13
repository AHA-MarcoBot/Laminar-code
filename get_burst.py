import numpy as np

TRACE_LENGTH = 10000

def load_data(path, maxlen=None, minlen=0, traces=0, dnn_type=None, openw=False):
    npzfile = np.load(path)
    data = npzfile["X"]
    labels = npzfile["y"]
    
    npzfile.close()
    return data, labels

def dir_to_burst(dir_data):
    burst_data = np.zeros(dir_data.shape)
    for i in range(len(dir_data)):
        k = 0
        dir_sign = dir_data[i][0]
        for j in range(TRACE_LENGTH):
            if dir_data[i][j] == 0:
                break
            if dir_data[i][j] != dir_sign:
                k += 1
                dir_sign = dir_data[i][j]
            burst_data[i][k] += dir_data[i][j]
    return burst_data

data, labels = load_data("./dataset/Closed World/tor_200w_2500tr.npz")
data = np.sign(data)

burst = dir_to_burst(data)

burst = burst[:, :2000]
np.savez('./dataset/Burst_Closed World/burst_tor_200w_2500tr.npz', data=burst, labels=labels)