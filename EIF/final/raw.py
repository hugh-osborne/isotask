import pylab
import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import imp
import datetime
from operator import add
from random import randint
import nimfa

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tkinter as tk
import struct
import itertools
import nolds
import nimfa
from nimfa.methods.seeding.nndsvd import Nndsvd
import os
import itertools
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy import signal
from scipy import stats
from sklearn.decomposition import NMF

def amplitude_normalization(x):
    x=x/np.max(x)
    return x

def butter_bandpass(lowcut,highcut,sampling_frequency,order=5):
    nyq = 0.5 *sampling_frequency
    low = lowcut/nyq
    high = highcut/nyq
    b,a = signal.butter (order,(low,high), btype = 'bandpass')
    return b,a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_lowpass(lowcut,sampling_frequency,order=5):
    nyq = 0.5 *sampling_frequency
    low = lowcut/nyq
    b,a = signal.butter (order,(low), btype = 'lowpass')
    return b,a

def butter_lowpass_filter(data, lowcut, fs, order=5):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


sim = numpy.loadtxt('ag_90_zero_raw.csv', delimiter=",").transpose()
exp = numpy.loadtxt('average_raw.csv', delimiter=",").transpose()

print(exp.shape)

fig,ax = plt.subplots(5,1,figsize=(5,8))
bx = ax
for i in range(5):
    # exp[i]=butter_bandpass_filter(exp[i],lowcut=20,highcut=450,fs=2000,order=2)
    # exp[i]=butter_lowpass_filter(abs(exp[i]),lowcut=20,fs=2000,order=2)
    # exp[i] = np.convolve(exp[i], np.ones((1000,))/1000, mode='same')
    rects1 = ax[i].plot(numpy.linspace(0,8.0,8/0.0005),((exp[i]*100000).tolist())[0:16000],color='#000000', linewidth=2)
    ax[i].tick_params(axis='x', bottom=False,labelbottom=(i == 4), labelsize=18)
    ax[i].xaxis.set_ticks(numpy.arange(0.0, 8.01, 8))
    ax[i].spines["top"].set_visible(False)
    #ax[i].set_xlim(0,8.5)
    ax[i].tick_params(axis='y', labelsize=20)
    bx[i] = ax[i].twinx()


for i in range(5):
	rects2 = bx[i].plot(numpy.linspace(0,8.0,8/0.002),(sim[i][0:16000]),color='red', linewidth=2)
	bx[i].tick_params(axis='y', bottom=(i == 4),labelbottom=(i == 4), labelsize=18)
	#bx[i].yaxis.set_ticks(numpy.arange(0.0, 50.01, 20))
	bx[i].spines["top"].set_visible(False)


plt.show()
