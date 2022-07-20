from scipy.io import loadmat
import numpy as np
import pickle
import scipy.io as sio
from scipy import signal
import pandas as pd
from pathlib import Path
from VMDHRBR import VMDHRBR
import pca_filter

def featureEnergy(meanData):
    #     输入1198*32*64*128
    for i in range(int(meanData.shape[0])):
        molecular = 0
        denominator = 0
        for j in range(int(meanData.shape[1])):
            molecular = meanData[i, j] * i + molecular
            denominator = meanData[i, j] + denominator
    return denominator



def featureMean(meanData):
    #     输入1198*32*64*128
    molecular = 0
    denominator = 0
    for i in range(int(meanData.shape[0])):
        for j in range(int(meanData.shape[1])):
            molecular = meanData[i, j] + molecular
    denominator = (meanData.shape[0] + 1)*(meanData.shape[1])
    mean = molecular / denominator
    return mean

def featureVar(meanData):
    fvar = np.var(abs(meanData))
    return fvar

def featureskew(meanData):
    a = []
    for i in range(21):
        a.append(abs(meanData[i]))
    a = np.concatenate(a, axis=0)
    fskew = pd.Series(a).skew()
    return fskew

def featurekurt(meanData):
    a = []
    for i in range(21):
        a.append(abs(meanData[i]))
    a = np.concatenate(a, axis=0)
    kurt = pd.Series(a).kurt()
    return kurt

def featurerms(meanData):
    a = []
    for i in range(21):
        a.append(abs(meanData[i]))
    a = np.concatenate(a, axis=0)
    rms = np.sqrt(np.mean(a ** 2))
    return rms

def feature18RD(data):
    featureE = featureEnergy(data)
    featureM = featureMean(data)
    featureR = featureVar(data)
    featureFs = featureskew(data)
    featureKu = featurekurt(data)
    featureRm = featurerms(data)
    return [featureE, featureM, featureR,  featureFs, featureKu, featureRm]

count = 0
labelY = np.ones(912)#ID
labelYY = np.ones(912)#pos
Data = []
heartrate = []
breathrate = []
for i in range(8):
    filename = 'E:/radar/20220401/ID/'+str(i+1)+'.mat'
    my_file = Path(filename)
    UWB = loadmat(my_file)['data']
    UWB = UWB[:, 0:400]
    UWB = pca_filter.p_f(UWB, 20, 0)
    max = [0] * UWB.shape[1]
    for i in range(UWB.shape[1]):
        max[i] = sum(UWB[:, i] * UWB[:, i])
    sig = UWB[:, np.argmax(max)]
    print(sig)
    out = VMDHRBR(sig)
    hr = int(round(out[0]))
    br = int(round(out[1]))
    heartrate.append(hr)
    breathrate.append(br)

for i in range(1):  # group
    for ii in range(8):  # ID
        for iii in range(800):  # time
            filename1 = 'E:/radar/20220401/slice/' + 'stand_' + str(ii + 1) + '_' + str(iii + 1) + '.mat'
            my_file1 = Path(filename)
            if my_file.exists():
                print(filename)
                UWB = loadmat(my_file1)['data']
                f18_1 = feature18RD(abs(UWB))
                f18_1.append(heartrate[ii])
                f18_1.append(breathrate[ii])
                Data.append(f18_1)
                labelY[count] = ii
                labelYY[count] = 0
                count = count + 1

            if not (my_file1.exists()):
                continue