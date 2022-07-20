from scipy.io import loadmat
import numpy as np
import pickle
import scipy.io as sio
from scipy import signal
import pandas as pd
from pathlib import Path
from VMDHRBR import VMDHRBR
import pca_filter

heartrate = []
breathrate = []
for i in range(8):
    for ii in range(400):
        filename = 'E:/radar/20220703/slice/' + str(i + 1) + '_' + str(ii + 1) + '.mat'
        my_file = Path(filename)
        UWB = loadmat(my_file)['data']
        max = [0] * UWB.shape[1]
        for iii in range(UWB.shape[1]):
            max[i] = sum(UWB[:, i] * UWB[:, i])
        sig = UWB[:, np.argmax(max)]
        #print(sig)
        out = VMDHRBR(sig)
        hr = int(round(out[0]))
        br = int(round(out[1]))
        heartrate.append(hr)
        breathrate.append(br)

sio.savemat('E:/radar/20220703/hr.mat', {'data': heartrate})
sio.savemat('E:/radar/20220703/br.mat', {'data': breathrate})
