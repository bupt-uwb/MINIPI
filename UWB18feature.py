from scipy.io import loadmat
import numpy as np
import pickle
import scipy.io as sio
from scipy import signal
import pandas as pd
from pathlib import Path
from VMDHRBR import VMDHRBR
import pca_filter



def Z_ScoreNormalization(x, mu, sigma):
    x = (x - mu) / sigma
    return x


def find2Section(data, size, interval):
    m, n = data.shape
    average_distance_weight = np.ones(n)
    for i in range(n):
        average_distance_weight[i] = np.sum(abs(data[10:, i]), axis=0)
    # print(average_distance_weight.shape)
    average_section_sum = np.ones(n)
    average_section_sum[0] = average_distance_weight[0]
    for ii in range(n - 1):
        average_section_sum[ii + 1] = average_section_sum[ii] + average_distance_weight[ii]
    # print(average_section_sum)
    average_section_weight = np.ones(n - size)
    for j in range(n - size):
        average_section_weight[j] = average_section_sum[j + size] - average_section_sum[j]

    num_peak_3 = signal.find_peaks(average_section_weight, distance=interval, width=5)  # prominences=0.1
    # print(num_peak_3[0])
    start1 = num_peak_3[0][0]
    start2 = num_peak_3[0][1]
    # print(average_section_weight)
    # print(start1,start2)
    return start1, start2



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
        a.append(abs(UWB[i]))
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

#
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
            filename2 = 'E:/radar/20220401/slice/' + 'sit_' + str(ii + 1) + '_' + str(iii + 1) + '.mat'
            # print(filename)
            my_file1 = Path(filename1)
            my_file2 = Path(filename2)
            if my_file1.exists():
                print(filename1)
                UWB = loadmat(my_file1)['data']
                f18_1 = feature18RD(abs(UWB))
                f18_1.append(heartrate[ii])
                f18_1.append(breathrate[ii])
                Data.append(f18_1)
                labelY[count] = ii
                labelYY[count] = 0
                count = count + 1

            if my_file2.exists():
                print(filename2)
                UWB = loadmat(my_file2)['data']
                f18_1 = feature18RD(abs(UWB))
                Data.append(f18_1)
                labelY[count] = ii
                labelYY[count] = 1
                count = count + 1

            if not (my_file1.exists() or my_file2.exists()):
                continue
with open('./UWB/UWB.data', 'wb') as filehandle:
    pickle.dump(Data, filehandle)
sio.savemat('./UWB/lableY.mat', {'data': labelY})
sio.savemat('./UWB/lableYY.mat', {'data': labelYY})
print(count)
