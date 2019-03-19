
"""
Created on Thu Feb  1 16:10:27 2018

@author: bsgjry
"""

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

def datafile_generator(wholefile):
        rf=[]
        vl=[]
        vm=[]
        st=[]
        bf=[]
        mg=[]
        ta=[]
        for idx in range(0,len(wholefile[:,1])):
            rf.append(struct.pack('<f',wholefile[idx,1]))
        for idx in range(0,len(wholefile[:,1])):
            vl.append(struct.pack('<f',wholefile[idx,2]))
        for idx in range(0,len(wholefile[:,1])):
            vm.append(struct.pack('<f',wholefile[idx,3]))
        for idx in range(0,len(wholefile[:,1])):
            st.append(struct.pack('<f',wholefile[idx,4]))
        for idx in range(0,len(wholefile[:,1])):
            bf.append(struct.pack('<f',wholefile[idx,5]))
        for idx in range(0,len(wholefile[:,1])):
            mg.append(struct.pack('<f',wholefile[idx,6]))
        for idx in range(0,len(wholefile[:,1])):
            ta.append(struct.pack('<f',wholefile[idx,7]))
        b_arr=[]
        c_arr=[]
        d_arr=[]
        e_arr=[]
        f_arr=[]
        g_arr=[]
        h_arr=[]
        for idx in range(0,len(rf)):
            b_arr.append(struct.unpack('<f',rf[idx]))
        for idx in range(0,len(vl)):
            c_arr.append(struct.unpack('<f',vl[idx]))
        for idx in range(0,len(vm)):
            d_arr.append(struct.unpack('<f',vm[idx]))
        for idx in range(0,len(st)):
            e_arr.append(struct.unpack('<f',st[idx]))
        for idx in range(0,len(bf)):
            f_arr.append(struct.unpack('<f',bf[idx]))
        for idx in range(0,len(mg)):
            g_arr.append(struct.unpack('<f',mg[idx]))
        for idx in range(0,len(ta)):
            h_arr.append(struct.unpack('<f',ta[idx]))
        b_arr=np.asarray((b_arr),dtype=np.float32)
        c_arr=np.asarray((c_arr),dtype=np.float32)
        d_arr=np.asarray((d_arr),dtype=np.float32)
        e_arr=np.asarray((e_arr),dtype=np.float32)
        f_arr=np.asarray((f_arr),dtype=np.float32)
        g_arr=np.asarray((g_arr),dtype=np.float32)
        h_arr=np.asarray((h_arr),dtype=np.float32)
        b_arr=b_arr.flatten()
        c_arr=c_arr.flatten()
        d_arr=d_arr.flatten()
        e_arr=e_arr.flatten()
        f_arr=f_arr.flatten()
        g_arr=g_arr.flatten()
        h_arr=h_arr.flatten()
        datafile1=np.column_stack((b_arr,c_arr,d_arr,e_arr,f_arr))
        return datafile1

def open_file():
    root = tk.Tk()
    root.withdraw()
    file_path = tk.filedialog.askopenfilename()
    wholefile=np.genfromtxt(file_path,delimiter=',',skip_header=1)
    return wholefile

def select_directory():
    root = tk.Tk()
    root.withdraw()
    directory = tk.filedialog.askdirectory()
    return directory

def windows(iterable, length=2, overlap=0):
    it = iter(iterable)
    results = list(itertools.islice(it, length))
    while len(results) == length:
        yield results
        results = results[length-overlap:]
        results.extend(itertools.islice(it, length-overlap))
    if results:
        yield results

def sampEn(x):
    window_length=200
    window_overlap=0
    stdDev=np.std(x)
    r=0.15*stdDev
    window_data=list(windows(x,length=window_length,overlap=window_overlap))
    window_data=np.transpose(np.asarray(window_data))
    out=np.zeros([int(len(x)/window_length),1])
    for window in range(0,len(out)):
        out[window]=nolds.sampen(data=window_data[:,window],emb_dim=2,tolerance=r)
    return out

def amplitude_thresholder(x):
    for column in range(1,np.shape(x)[1]):
        baseline=np.mean(x[0:2000,column])
        x[:,column]=x[:,column]-baseline
    return x

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

def PCA_calculator(x):
    #Mean normalization  is a requirement for this algorithm
    if type(x) == np.ma.core.MaskedArray:
        x=x.filled(0)
    X_std = StandardScaler().fit_transform(x)
    pca=PCA(n_components=0.8,svd_solver='full')
    transformed_data = pca.fit_transform(X_std)
    vector=pca.components_
    print('Components kept : %s'%str(pca.n_components_))
    print('Explained variance ratio : %s' %str(pca.explained_variance_ratio_))
    return vector,singular_values

def NMF_calculator(x,rank):
    nmf = nimfa.Nmf(abs(x), seed='nndsvd', rank=rank, max_iter=5000)
    nmf_fit = nmf()
    print('Evar: %5.4f' % nmf_fit.fit.evar())
    vector=np.asarray(nmf_fit.basis())
    synergy_coeff=nmf_fit.fit.coef()
    residuals=np.asarray(nmf_fit.fit.residuals())
    vaf_muscle=np.zeros([1,x.shape[1]])
    for column in range(0,x.shape[1]):
        result = 1-(sum(residuals[:,column]**2)/sum(x[:,column]**2))
        vaf_muscle[0,column] = result
    return vector,vaf_muscle,synergy_coeff

def pearsons_correlation_muscles(directory):
    muscle_correlation = []
    files=[]
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            files.append(file)
    files=list(itertools.combinations(files,2))
    for combination in files:
        wholefile1=np.genfromtxt(os.path.join(directory, combination[0]),delimiter=',',skip_header=1)
        wholefile2=np.genfromtxt(os.path.join(directory, combination[1]),delimiter=',',skip_header=1)
        datafile1=datafile_generator(wholefile1)
        datafile2=datafile_generator(wholefile2)
        for column in range(0,datafile1.shape[1]):
            datafile1[:,column]=butter_bandpass_filter(datafile1[:,column],lowcut=20,highcut=450,fs=2000,order=2)
            datafile1[:,column]=butter_lowpass_filter(abs(datafile1[:,column]),lowcut=9,fs=2000,order=2)
            datafile1[:,column]=amplitude_normalization(datafile1[:,column])
        for column in range(0,datafile2.shape[1]):
            datafile2[:,column]=butter_bandpass_filter(datafile2[:,column],lowcut=20,highcut=450,fs=2000,order=2)
            datafile2[:,column]=butter_lowpass_filter(abs(datafile2[:,column]),lowcut=9,fs=2000,order=2)
            datafile2[:,column]=amplitude_normalization(datafile2[:,column])
        for column in range(0,datafile1.shape[1]):
            muscle_correlation.append(stats.pearsonr(datafile1[:,column], datafile2[:,column]))
    return muscle_correlation

def cosine_correlation_muscles(directory):
    muscle_correlation = []
    files=[]
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            files.append(file)
    files=list(itertools.combinations(files,2))
    for combination in files:
        wholefile1=np.genfromtxt(os.path.join(directory, combination[0]),delimiter=',',skip_header=1)
        wholefile2=np.genfromtxt(os.path.join(directory, combination[1]),delimiter=',',skip_header=1)
        datafile1=datafile_generator(wholefile1)
        datafile2=datafile_generator(wholefile2)
        for column in range(0,datafile1.shape[1]):
            datafile1[:,column]=butter_bandpass_filter(datafile1[:,column],lowcut=20,highcut=450,fs=2000,order=2)
            datafile1[:,column]=butter_lowpass_filter(abs(datafile1[:,column]),lowcut=9,fs=2000,order=2)
            datafile1[:,column]=amplitude_normalization(datafile1[:,column])
        for column in range(0,datafile2.shape[1]):
            datafile2[:,column]=butter_bandpass_filter(datafile2[:,column],lowcut=20,highcut=450,fs=2000,order=2)
            datafile2[:,column]=butter_lowpass_filter(abs(datafile2[:,column]),lowcut=9,fs=2000,order=2)
            datafile2[:,column]=amplitude_normalization(datafile2[:,column])
        for column in range(0,datafile1.shape[1]):
            muscle_correlation.append(pairwise.cosine_similarity(((np.ravel(datafile1[:,column])),np.ravel((datafile2[:,column]))))[0][1])
    return muscle_correlation

def NMF_calculator_shared(x,start_vector,start_coeff,rank):
    nmf = nimfa.Nmf(abs(x), seed=None, W=start_coeff, H=start_vector, rank=rank, max_iter=500)
    nmf_fit = nmf()
    print('Evar: %5.4f' % nmf_fit.fit.evar())
    vector=np.asarray(nmf_fit.basis())
    synergy_coeff=nmf_fit.fit.coef()
    residuals=np.asarray(nmf_fit.fit.residuals())
    vaf_muscle=np.zeros([1,x.shape[1]])
    for column in range(0,x.shape[1]):
        result = 1-(sum(residuals[:,column]**2)/sum(x[:,column]**2))
        vaf_muscle[0,column] = result
    return vector,vaf_muscle,synergy_coeff

def synergy_analysis_shared(directory):
    time_series_length = 20000
    t_start = 18000
    t_end = 38000
    num_angles = 4
    num_coefficients = 5
    vector = np.zeros([time_series_length*num_angles,0])
    vaf_muscle = np.zeros([0,num_coefficients])
    synergy_coeff = np.zeros([0,num_coefficients])
    synergy1_vector_correlation = []
    synergy1_coeff_correlation = []
    files=[]
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            files.append(file)
    datafile_full = np.zeros([0,num_coefficients])
    for file in files:
        wholefile=np.genfromtxt(os.path.join(directory, file),delimiter=',',skip_header=1)
        df1=datafile_generator(wholefile)
        datafile1 = df1[t_start:t_end]
        for column in range(0,datafile1.shape[1]):
            datafile1[:,column]=butter_bandpass_filter(datafile1[:,column],lowcut=20,highcut=450,fs=2000,order=2)
            datafile1[:,column]=butter_lowpass_filter(abs(datafile1[:,column]),lowcut=20,fs=2000,order=2)
            datafile1[:,column] = np.convolve(datafile1[:,column], np.ones((500,))/500, mode='same')
            datafile1[:,column]=amplitude_normalization(datafile1[:,column])

        print("datafile shape:")
        print(datafile1.shape)
        datafile_full = np.append(datafile_full, datafile1, axis=0)


    print('FULL LENGTH : ')
    print(datafile_full.shape)

    start_coeff = [0,0,0,0,0]
    start_vector =
    a,b,c=NMF_calculator_shared(datafile_full,rank=2)

    plt.figure()

    a = np.transpose(a)
    plt.subplot(411)
    plt.bar(index,np.ravel(c[0].tolist()[0]),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))
    plt.subplot(412)
    plt.bar(index,np.ravel(c[1].tolist()[0]),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))
    plt.subplot(413)
    plt.plot(a[0].tolist())
    plt.subplot(414)
    plt.plot(a[1].tolist())
    plt.show()


def synergy_analysis(directory):
    vector = np.zeros([20000,0])
    stuff = np.zeros([20000,0])
    vaf_muscle = np.zeros([0,5])
    synergy_coeff = np.zeros([0,5])
    synergy1_vector_correlation = []
    synergy2_vector_correlation = []
    synergy1_coeff_correlation = []
    synergy2_coeff_correlation = []
    #directory = select_directory()
    files=[]
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            files.append(file)
    num_actions = 1
    for file in files:
        wholefile=np.genfromtxt(os.path.join(directory, file),delimiter=',',skip_header=1)
        df1=datafile_generator(wholefile)
        for index in range(num_actions):
            datafile1 = df1[18000:38000]
            # plt.figure()
            # plt.subplot(511)
            for column in range(0,datafile1.shape[1]):
                datafile1[:,column]=butter_bandpass_filter(datafile1[:,column],lowcut=20,highcut=450,fs=2000,order=2)
                datafile1[:,column]=butter_lowpass_filter(abs(datafile1[:,column]),lowcut=20,fs=2000,order=2)
                datafile1[:,column] = np.convolve(datafile1[:,column], np.ones((3000,))/3000, mode='same')
                datafile1[:,column]=amplitude_normalization(datafile1[:,column])

                # plt.subplot(511 + column)
                # plt.plot(datafile1[:,column].tolist())

                # datafile1[0:1000,column] = np.zeros(1000)
                # datafile1[200000:20000,column] = np.zeros(1000)
            stuff = np.append(stuff, datafile1, axis=1)
            print(stuff.shape)
            a,b,c=NMF_calculator(datafile1,rank=2)
            #
            index = np.arange(5)
            names=("RF","VL","VM","ST","BF")
            percentage = ("0","25","50","75","100")

            # plt.figure()
            #
            #
            # #
            # a = np.transpose(a)
            # print(len(a[0].tolist()))
            # print(b.shape)
            # print(c)
            # plt.subplot(411)
            # plt.bar(index,np.ravel(c[0].tolist()[0]),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))
            # plt.subplot(412)
            # plt.bar(index,np.ravel(c[1].tolist()[0]),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))
            # plt.subplot(413)
            # plt.plot(a[0].tolist())
            # plt.subplot(414)
            # plt.plot(a[1].tolist())
            # plt.show()
            # #
            # a = np.transpose(a)

            # if vector.shape[0] != amplitude_normalization(a).shape[0]:
            #     continue
            vector=np.append(vector,a,axis=1)
            print(vector.shape)
            # vector=amplitude_normalization(vector)
            vaf_muscle=np.append(vaf_muscle,b,axis=0)
            print(vaf_muscle.shape)
            synergy_coeff=np.append(synergy_coeff,c,axis=0)
            print(synergy_coeff.shape)
    vector=np.transpose(vector)
    stuff=np.transpose(stuff)
    synergy1_vector_combinations = list(itertools.combinations(vector[0:num_actions*30:2,:],2))
    synergy2_vector_combinations = list(itertools.combinations(vector[1:num_actions*30:2,:],2))
    synergy1_coeff_combinations = list(itertools.combinations(synergy_coeff[0:num_actions*30:2],2))
    synergy2_coeff_combinations = list(itertools.combinations(synergy_coeff[1:num_actions*30:2],2))
    """
    for combination in synergy1_vector_combinations:
          synergy1_vector_correlation.append(stats.pearsonr(combinaa,b,c=NMF_calculator(datafile1,rank=2)tion[0],combination[1]))
    for combination in synergy2_vector_combinations:
          synergy2_vector_correlation.append(stats.pearsonr(combination[0],combination[1]))
    for combination in synergy1_coeff_combinations:
          synergy1_coeff_correlation.append(stats.pearsonr(np.ravel(combination[0]),np.ravel(combination[1])))
    for combination in synergy2_coeff_combinations:
          synergy2_coeff_correlation.append(stats.pearsonr(np.ravel(combination[0]),np.ravel(combination[1])))
    """
    for combination in synergy1_vector_combinations:
          synergy1_vector_correlation.append(pairwise.cosine_similarity((combination[0],combination[1]))[0][1])
    for combination in synergy2_vector_combinations:
          synergy2_vector_correlation.append(pairwise.cosine_similarity((combination[0],combination[1]))[0][1])
    for combination in synergy1_coeff_combinations:
          synergy1_coeff_correlation.append(pairwise.cosine_similarity((combination[0]),(combination[1]))[0])
    for combination in synergy2_coeff_combinations:
          synergy2_coeff_correlation.append(pairwise.cosine_similarity((combination[0]),(combination[1]))[0])
    synergy1_vector_correlation=np.asarray(synergy1_vector_correlation)
    synergy2_vector_correlation=np.asarray(synergy2_vector_correlation)
    synergy1_coeff_correlation=np.asarray(synergy1_coeff_correlation)
    synergy2_coeff_correlation=np.asarray(synergy2_coeff_correlation)
    return vector,vaf_muscle,synergy_coeff,synergy1_vector_correlation,synergy2_vector_correlation,synergy1_coeff_correlation,synergy2_coeff_correlation,stuff

def interpolator(x):
    for column in range(0,5):
        xp=(np.where(x[:,column]>(np.max(x[:,column])*0.15)))[0]
        yp=x[(np.where(x[:,column]>(np.max(x[:,column])*0.15)))[0],column]
        x[:,column]=np.interp(range(0,20000),xp,yp)
        x[:,column]=x[:,column]-x[0,column]
    return x

def elbow_calculator(x):
    for idx in range(1,len(x)):
        print((KMeans(n_clusters=idx,random_state=0).fit(x)).inertia_)


directory0_1="S1_seperated by angle/0 degrees"
directory20_1="S1_seperated by angle/20 degrees"
directory60_1="S1_seperated by angle/60 degrees"
directory90_1="S1_seperated by angle/90 degrees"

directory0_2="S2_seperated by angle/0 degrees"
directory20_2="S2_seperated by angle/20 degrees"
directory60_2="S2_seperated by angle/60 degrees"
directory90_2="S2_seperated by angle/90 degrees"
directory=directory0_1
"""
directory_all=select_directory()
directory0_1=select_directory()
directory20_1=select_directory()
directory60_1=select_directory()
directory90_1=select_directory()

directory0_2=select_directory()        average_muscle_correlation[2,muscle]=np.mean(muscle_correlation60[muscle:-1:5,0])
for muscle in range(0,5):
        average_muscle_correlation[3,muscle]=np.mean(muscle_correlation90[muscle:-1:5,0])
plt.bar(index,np.ravel(synergy_coeff0_1),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))
directory20_2=select_directory()
directory60_2=select_directory()
directory90_2=select_directory()
directory=directory0_1


vector,vaf_muscle,synergy_coeff,synergy1_vector_correlation,synergy2_vector_correlation,synergy1_coeff_correlation,synergy2_coeff_correlation=synergy_analysis(directory=directory_all)
average_synergy1_coeff=np.zeros([1,5])
average_synergy2_coeff=np.zeros([1,5])
for coeff in range(0,5):
    average_synergy1_coeff[0,coeff]=np.mean(synergy_coeff[0:-1:2,coeff])
for coeff in range(0,5):
    average_synergy2_coeff[0,coeff]=np.mean(synergy_coeff[1:-1:2,coeff])


muscle_correlation0=np.asarray(pearsons_correlation_musclestuffs(directory=directory0))
muscle_correlation20=np.asarray(pearsons_correlation_muscles(directory=directory20))
muscle_correlation60=np.asarray(pearsons_correlation_muscles(directory=directory60))
muscle_correlation90=np.asarray(pearsons_correlation_muscles(directory=directory90))

muscle_cosine_correlation0=np.asarray(cosine_correlation_muscles(directory=directory0))
muscle_cosine_correlation20=np.asarray(cosine_correlation_muscles(directory=directory0))
muscle_cosine_correlation60=np.asarray(cosine_correlation_muscles(directory=directory0))
muscle_cosine_correlation90=np.asarray(cosine_correlation_muscles(directory=directory0))

average_cosine_muscle_correlation=np.zeros([4,5])
for muscle in range(0,5):
        average_cosine_muscle_correlation[0,muscle]=np.mean(muscle_cosine_correlation0[muscle:-1:5])
for muscle in range(0,5):
        average_cosine_muscle_correlation[1,muscle]=np.mean(muscle_cosine_correlation20[muscle:-1:5])
for muscle in range(0,5):
        average_cosine_muscle_correlation[2,muscle]=np.mean(muscle_cosine_correlation60[muscle:-1:5])
for muscle in range(0,5):
        average_cosine_muscle_correlation[3,muscle]=np.mean(muscle_cosine_correlation90[muscle:-1:5])

average_muscle_correlation=np.zeros([4,5])
for muscle in range(0,5):
        average_muscle_correlation[0,muscle]=np.mean(muscle_correlation0[muscle:-1:5,0])
for muscle in range(0,5):
        average_muscle_correlation[1,muscle]=np.mean(muscle_correlation20[muscle:-1:5,0])
for muscle in range(0,5):where(labels==1),0]),xs=range(1,8),c="red",marker="o",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(norm[np.where(labels==1),1]),xs=range(1,8),c="blue",marker="v",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(norm[np.where(labels==1),2]),xs=range(1,8),c="green",marker="s",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(norm[np.where(labels==1),3]),xs=range(1,8),c="yellow",marker="*",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(norm[np.wher
        average_muscle_correlation[2,muscle]=np.mean(muscle_correlation60[muscle:-1:5,0])
for muscle in range(0,5):
        average_muscle_correlation[3,muscle]=np.mean(muscle_correlation90[muscle:-1:5,0])
plt.bar(index,np.ravel(synergy_coeff0_1),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))

"""
# vector0_1,vaf_muscle0_1,synergy_coeff0_1,synergy1_vector_correlation0_1,synergy2_vector_correlation0_1,synergy1_coeff_correlation0_1,synergy2_coeff_correlation0_1,stuff0_1=synergy_analysis(directory=directory0_1)
# vector20_1,vaf_muscle20_1,synergy_coeff20_1,synergy1_vector_correlation20_1,synergy2_vector_correlation20_1,synergy1_coeff_correlation20_1,synergy2_coeff_correlation20_1=synergy_analysis(directory=directory20_1)
# vector60_1,vaf_muscle60_1,synergy_coeff60_1,synergy1_vector_correlation60_1,synergy2_vector_correlation60_1,synergy1_coeff_correlation60_1,synergy2_coeff_correlation60_1=synergy_analysis(directory=directory60_1)
# vector90_1,vaf_muscle90_1,synergy_coeff90_1,synergy1_vector_correlation90_1,synergy2_vector_correlation90_1,synergy1_coeff_correlation90_1,synergy2_coeff_correlation90_1=synergy_analysis(directory=directory90_1)

synergy_analysis_shared(directory="S1_seperated by angle/p1")
# vector0_2,vaf_muscle0_2,synergy_coeff0_2,synergy1_vector_correlation0_2,synergy2_vector_correlation0_2,synergy1_coeff_correlation0_2,synergy2_coeff_correlation0_2=synergy_analysis(directory=directory0_2)
# vector20_2,vaf_muscle20_2,synergy_coeff20_2,synergy1_vector_correlation20_2,synergy2_vector_correlation20_2,synergy1_coeff_correlation20_2,synergy2_coeff_correlation20_2=synergy_analysis(directory=directory20_2)
# vector60_2,vaf_muscle60_2,synergy_coeff60_2,synergy1_vector_correlation60_2,synergy2_vector_correlation60_2,synergy1_coeff_correlation60_2,synergy2_coeff_correlation60_2=synergy_analysis(directory=directory60_2)
# vector90_2,vaf_muscle90_2,synergy_coeff90_2,synergy1_vector_correlation90_2,synergy2_vector_correlation90_2,synergy1_coeff_correlation90_2,synergy2_coeff_correlation90_2=synergy_analysis(directory=directory90_2)

average_synergy1_coeff_correlation_1=np.zeros([4,1])
average_synergy1_coeff_correlation_1[0,0]=np.mean(synergy1_coeff_correlation0_1[:,0])
# average_synergy1_coeff_correlation_1[1,0]=np.mean(synergy1_coeff_correlation20_1[:,0])
# average_synergy1_coeff_correlation_1[2,0]=np.mean(synergy1_coeff_correlation60_1[:,0])
# average_synergy1_coeff_correlation_1[3,0]=np.mean(synergy1_coeff_correlation90_1[:,0])

average_synergy2_coeff_correlation_1=np.zeros([4,1])
average_synergy2_coeff_correlation_1[0,0]=np.mean(synergy2_coeff_correlation0_1[:,0])
# average_synergy2_coeff_correlation_1[1,0]=np.mean(synergy2_coeff_correlation20_1[:,0])
# average_synergy2_coeff_correlation_1[2,0]=np.mean(synergy2_coeff_correlation60_1[:,0])
# average_synergy2_coeff_correlation_1[3,0]=np.mean(synergy2_coeff_correlation90_1[:,0])

average_synergy1_vector_correlation_1=np.zeros([4,1])
average_synergy1_vector_correlation_1[0,0]=np.mean(synergy1_vector_correlation0_1)
# average_synergy1_vector_correlation_1[1,0]=np.mean(synergy1_vector_correlation20_1)
# average_synergy1_vector_correlation_1[2,0]=np.mean(synergy1_vector_correlation60_1)
# average_synergy1_vector_correlation_1[3,0]=np.mean(synergy1_vector_correlation90_1)

average_synergy2_vector_correlation_1=np.zeros([4,1])
average_synergy2_vector_correlation_1[0,0]=np.mean(synergy2_vector_correlation0_1)
# average_synergy2_vector_correlation_1[1,0]=np.mean(synergy2_vector_correlation20_1)
# average_synergy2_vector_correlation_1[2,0]=np.mean(synergy2_vector_correlation60_1)
# average_synergy2_vector_correlation_1[3,0]=np.mean(synergy2_vector_correlation90_1)
#
# average_synergy1_coeff_correlation_2=np.zeros([4,1])
# average_synergy1_coeff_correlation_2[0,0]=np.mean(synergy1_coeff_correlation0_2[:,0])
# average_synergy1_coeff_correlation_2[1,0]=np.mean(synergy1_coeff_correlation20_2[:,0])
# average_synergy1_coeff_correlation_2[2,0]=np.mean(synergy1_coeff_correlation60_2[:,0])
# average_synergy1_coeff_correlation_2[3,0]=np.mean(synergy1_coeff_correlation90_2[:,0])
#
# average_synergy2_coeff_correlation_2=np.zeros([4,1])
# average_synergy2_coeff_correlation_2[0,0]=np.mean(synergy2_coeff_correlation0_2[:,0])
# average_synergy2_coeff_correlation_2[1,0]=np.mean(synergy2_coeff_correlation20_2[:,0])
# average_synergy2_coeff_correlation_2[2,0]=np.mean(synergy2_coeff_correlation60_2[:,0])
# average_synergy2_coeff_correlation_2[3,0]=np.mean(synergy2_coeff_correlation90_2[:,0])
#
# average_synergy1_vector_correlation_2=np.zeros([4,1])
# average_synergy1_vector_correlation_2[0,0]=np.mean(synergy1_vector_correlation0_2)
# average_synergy1_vector_correlation_2[1,0]=np.mean(synergy1_vector_correlation20_2)
# average_synergy1_vector_correlation_2[2,0]=np.mean(synergy1_vector_correlation60_2)
# average_synergy1_vector_correlation_2[3,0]=np.mean(synergy1_vector_correlation90_2)
#
# average_synergy2_vector_correlation_2=np.zeros([4,1])
# average_synergy2_vector_correlation_2[0,0]=np.mean(synergy2_vector_correlation0_2)
# average_synergy2_vector_correlation_2[1,0]=np.mean(synergy2_vector_correlation20_2)
# average_synergy2_vector_correlation_2[2,0]=np.mean(synergy2_vector_correlation60_2)
# average_synergy2_vector_correlation_2[3,0]=np.mean(synergy2_vector_correlation90_2)

# norm=preprocessing.scale(synergy_coeff0_1[0:30:2])
# norm1=preprocessing.scale(synergy_coeff0_1[1:30:2])
# a=np.append(norm,norm1,axis=1)
#
# combinations=list(itertools.combinations(range(1,len(np.ravel(a[0]))),2))
# for combination in combinations:
#     print(stats.levene(np.ravel(a[:,combination[0]]),np.ravel(a[:,combination[1]]))[1])
#
# elbow_calculator(a)
# kmeans = KMeans(n_clusters=2)
# kmeans.fit(a)
# kmeans.predict(a)
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
#
# elbow_calculator(norm)
# kmeans = KMeans(n_clusters=2)
# kmeans.fit(norm)
# kmeans.predict(norm)synergy_coeff0_1
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
#
# elbow_calculator(norm1)
# kmeans1 = KMeans(n_clusters=2)
# kmeans.fit(norm1)
# kmeans.predict(norm1)
# labels1 = kmeans.labels_
# centroids = kmeans.cluster_centers_
#
# fig = plt.figure(figsize=(18,7))
# ax = fig.add_subplot(111,projection='3d')
# ax.set_xticks(range(1,15))
#
# ax.scatter(ys=np.ravel(a[np.where(labels==0),0]),xs=range(1,1+len(np.ravel(a[np.where(labels==0),0]))),zs=20,c="red",marker="o",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(a[np.where(labels==0),1]),xs=range(1,1+len(np.ravel(a[np.where(labels==0),0]))),zs=20,c="blue",marker="v",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(a[np.where(labels==0),2]),xs=range(1,1+len(np.ravel(a[np.where(labels==0),0]))),zs=20,c="green",marker="s",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(a[np.where(labels==0),3]),xs=range(1,1+len(np.ravel(a[np.where(labels==0),0]))),zs=20,c="yellow",marker="*",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(a[np.where(labels==0),4]),xs=range(1,1+len(np.ravel(a[np.where(labels==0),0]))),zs=20,c="purple",marker="p",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(a[np.where(labels==0),5]),xs=range(1,1+len(np.ravel(a[np.where(labels==0),0]))),zs=35,c="black",marker="o",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(a[np.where(labels==0),6]),xs=range(1,1+len(np.ravel(a[np.where(labels==0),0]))),zs=35,c="pink",marker="v",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(a[np.where(labels==0),7]),xs=range(1,1+len(np.ravel(a[np.where(labels==0),0]))),zs=35,c="gold",marker="s",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(a[np.where(labels==0),8]),xs=range(1,1+len(np.ravel(a[np.where(labels==0),0]))),zs=35,c="darkgreen",marker="*",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(a[np.where(labels==0),9]),xs=range(1,1+len(np.ravel(a[np.where(labels==0),0]))),zs=35,c="silver",marker="p",edgecolor='k',s=300)
#for muscle in range(0,5):
#         average_muscle_correlation[1,muscle]=np.mean(muscle_correlation20[muscle:-1:5,0])
# for muscle in range(0,5):where(labels==1),0]),xs=range(1,8),c="red",marker="o",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(a[np.where(labels==1),0]),xs=range(1,1+len(np.ravel(a[np.where(labels==1),0]))),zs=1,c="red",marker="o",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(a[np.where(labels==1),1]),xs=range(1,1+len(np.ravel(a[np.where(labels==1),0]))),zs=1,c="blue",marker="v",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(a[np.where(labels==1),2]),xs=range(1,1+len(np.ravel(a[np.where(labels==1),0]))),zs=1,c="green",marker="s",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(a[np.where(labels==1),3]),xs=range(1,1+len(np.ravel(a[np.where(labels==1),0]))),zs=1,c="yellow",marker="*",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(a[np.where(labels==1),4]),xs=range(1,1+len(np.ravel(a[np.where(labels==1),0]))),zs=1,c="purple",marker="p",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(a[np.where(labels==1),5]),xs=range(1,1+len(np.ravel(a[np.where(labels==1),0]))),zs=15,c="black",marker="o",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(a[np.where(labels==1),6]),xs=range(1,1+len(np.ravel(a[np.where(labels==1),0]))),zs=15,c="pink",marker="v",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(a[np.where(labels==1),7]),xs=range(1,1+len(np.ravel(a[np.where(labels==1),0]))),zs=15,c="gold",marker="s",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(a[np.where(labels==1),8]),xs=range(1,1+len(np.ravel(a[np.where(labels==1),0]))),zs=15,c="darkgreen",marker="*",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(a[np.where(labels==1),9]),xs=range(1,1+len(np.ravel(a[np.where(labels==1),0]))),zs=15,c="silver",marker="p",edgecolor='k',s=300)
#
#
# ax.scatter(ys=np.ravel(norm[np.where(labels==1),0]),xs=range(1,8),c="red",marker="o",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(norm[np.where(labels==1),1]),xs=range(1,8),c="blue",marker="v",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(norm[np.where(labels==1),2]),xs=range(1,8),c="green",marker="s",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(norm[np.where(labels==1),3]),xs=range(1,8),c="yellow",marker="*",edgecolor='k',s=300)
# ax.scatter(ys=np.ravel(norm[np.where(labels==1),4]),xs=range(1,8),c="purple",marker="p",edgecolor='k',s=300)
#
# fig = plt.figure(figsize=(18,7))
# ax = fig.add_subplot(111)
# ax.set_xticks(range(1,15))
#for muscle in range(0,5):
#         average_muscle_correlation[1,muscle]=np.mean(muscle_correlation20[muscle:-1:5,0])
# for muscle in range(0,5):where(labels==1),0]),xs=range(1,8),c="red",marker="o",edgecolor='k',s=300)
# ax.scatter(y=np.ravel(norm[np.where(labels==0),0]),x=range(1,8),c="red",marker="o",edgecolor='k',s=300)
# ax.scatter(y=np.ravel(norm[np.where(labels==0),1]),x=range(1,8),c="blue",marker="v",edgecolor='k',s=300)
# ax.scatter(y=np.ravel(norm[np.where(labels==0),2]),x=range(1,8),c="green",marker="s",edgecolor='k',s=300)
# ax.scatter(y=np.ravel(norm[np.where(labels==0),3]),x=range(1,8),c="yellow",marker="*",edgecolor='k',s=300)
# ax.scatter(y=np.ravel(norm[np.where(labels==0),4]),x=range(1,8),c="purple",marker="p",edgecolor='k',s=300)
#
# ax.scatter(y=np.ravel(norm[np.where(labels==1),0]),x=range(1,8),c="red",marker="o",edgecolor='k',s=300)
# ax.scatter(y=np.ravel(norm[np.where(labels==1),1]),x=range(1,8),c="blue",marker="v",edgecolor='k',s=300)
# ax.scatter(y=np.ravel(norm[np.where(labels==1),2]),x=range(1,8),c="green",marker="s",edgecolor='k',s=300)
# ax.scatter(y=np.ravel(norm[np.where(labels==1),3]),x=range(1,8),c="yellow",marker="*",edgecolor='k',s=300)
# ax.scatter(y=np.ravel(norm[np.where(labels==1),4]),x=range(1,8),c="purple",marker="p",edgecolor='k',s=300)
#
# ax.scatter(y=np.ravel(norm[np.where(labels==1),0]),x=range(1,8),c="red",marker="o",edgecolor='k',s=300)
# ax.scatter(y=np.ravel(norm[np.where(labels==1),1]),x=range(1,8),c="blue",marker="v",edgecolor='k',s=300)
# ax.scatter(y=np.ravel(norm[np.where(labels==1),2]),x=range(1,8),c="green",marker="s",edgecolor='k',s=300)
# ax.scatter(y=np.ravel(norm[np.where(labels==1),3]),x=range(1,8),c="yellow",marker="*",edgecolor='k',s=300)
# ax.scatter(y=np.ravel(norm[np.where(labels==1),4]),x=range(1,8),c="purple",marker="p",edgecolor='k',s=300)

"""
What do i want these graphs to demonstrate?
I want them to show the changes in co-eff recruitment as angle and position changes. This is most apparent
in synergy 2 as this shows the greatest change with angle. I will need to show statistically that the
coeff values are different at different angles. Difference in overall values not just average. Don't have
an independent variable for this though. Show the relationship between values and angle? Relationship
is non-linear for the most part, making correlation difficult.

Why?
Coeff is time independent and therefore reflects synergy changes across the whole activity better than
changes in vector. The changes in recruitment will demonstrate that different angles will result in different
movements.

Tomorrow: Do correlative tests, try and get a general linear model, using the biarticular muscles value as the
dependent variable. (BF and RF). Probably need to go over to R as the python module is a bit arcane.
"""


names=("RF","VL","VM","ST","BF")
# x_scaled=preprocessing.scale(synergy_coeff0_1)

# elbow_calculator(x_scaled[0:30:2])
# kmeans = KMeans(n_clusters=2)
# kmeans.fit(x_scaled[0:30:2])
# kmeans.predict(x_scaled[0:30:2])
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
# fig = plt.figure(figsize=(18,7))
# ax = fig.add_subplot(111)
# ax.scatter(y=np.ravel(x_scaled[0:30:2,0]),x=range(1,15),c=labels.astype(np.float),marker="o",edgecolor='k',s=300,alpha=0.5)
# ax.scatter(y=np.ravel(x_scaled[0:30:2,1]),x=range(1,15),c=labels.astype(np.float),marker="v",edgecolor='k',s=300,alpha=0.5)
# ax.scatter(y=np.ravel(x_scaled[0:30:2,2]),x=range(1,15),c=labels.astype(np.float),marker="s",edgecolor='k',s=300,alpha=0.5)
# ax.scatter(y=np.ravel(x_scaled[0:30:2,3]),x=range(1,15),c=labels.astype(np.float),marker="*",edgecolor='k',s=300,alpha=0.5)
# ax.scatter(y=np.ravel(x_scaled[0:30:2,4]),x=range(1,15),c=labels.astype(np.float),marker="p",edgecolor='k',s=300,alpha=0.5)
#
# elbow_calculator(synergy_coeff0_1[0:30:2])
# kmeans = KMeans(n_clusters=2)
# kmeans.fit(synergy_coeff0_1[0:30:2])
# kmeans.predict(synergy_coeff0_1[0:30:2])
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
# fig = plt.figure(figsize=(18,7))
# ax = fig.add_subplot(1d=preprocessing.scale(synergy_coeff0_1)11)
# ax.scatter(y=np.ravel(synergy_coeff0_1[0:30:2,0]),x=range(1,14),c=labels.astype(np.float),marker="o",edgecolor='k',s=300,alpha=0.5)
# ax.scatter(y=np.ravel(synergy_coeff0_1[0:30:2,1]),x=range(1,14),c=labels.astype(np.float),marker="v",edgecolor='k',s=300,alpha=0.5)
# ax.scatter(y=np.ravel(synergy_coeff0_1[0:30:2,2]),x=range(1,14),c=labels.astype(np.float),marker="s",edgecolor='k',s=300,alpha=0.5)
# ax.scatter(y=np.ravel(synergy_coeff0_1[0:30:2,3]),x=range(1,14),c=labels.astype(np.float),marker="*",edgecolor='k',s=300,alpha=0.5)
# ax.scatter(y=np.ravel(synergy_coeff0_1[0:30:2,4]),x=range(1,14),c=labels.astype(np.float),marker="p",edgecolor='k',s=300,alpha=0.5)
#
# elbow_calculator(synergy_coeff0_1[1:30:2])
# kmeans1 = KMeans(n_clusters=4)
# kmeans1.fit(synergy_coeff0_1[1:30:2])
# kmeans1.predict(synergy_coeff0_1[1:30:2])
# labels1 = kmeans1.labels_25
# centroids = kmeans1.cluster_centers_
# fig = plt.figure(figsize=(18,7))
# ax = fig.add_subplot(111)
# ax.scatter(y=np.ravel(synergy_coeff0_1[1:30:2,0]),x=range(1,15),c=labels1.astype(np.float),marker="o", edgecolor='k',s=300,alpha=0.5)
# ax.scatter(y=np.ravel(synergy_coeff0_1[1:30:2,1]),x=range(1,15),c=labels1.astype(np.float),marker="v",  edgecolor='k',s=300,alpha=0.5)
# ax.scatter(y=np.ravel(synergy_coeff0_1[1:30:2,2]),x=range(1,15),c=labels1.astype(np.float),marker="s",  edgecolor='k',s=300,alpha=0.5)
# ax.scatter(y=np.ravel(synergy_coeff0_1[1:30:2,3]),x=range(1,15),c=labels1.astype(np.float),marker="*",  edgecolor='k',s=300,alpha=0.5)
# ax.scatter(y=np.ravel(synergy_coeff0_1[1:30:2,4]),x=range(1,15),c=labels1.astype(np.float),marker="p",  edgecolor='k',s=300,alpha=0.5)
#
#
# elbow_calculator(synergy_coeff0_1[0:30:2])
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(synergy_coeff0_1[0:30:2])
# kmeans.predict(synergy_coeff0_1[0:30:2])
# labels = kmeans.labels_
#
# elbow_calculator(synergy_coeff20_1[0:30:2])
# kmeans1 = KMeans(n_clusters=3)
# kmeans1.fit(synergy_coeff20_1[0:30:2])
# kmeans1.predict(synergy_coeff20_1[0:30:2])
# labels1 = kmeans1.labels_
#
# elbow_calculator(synergy_coeff60_1[0:30:2])
# kmeans2 = KMeans(n_clusters=3)
# kmeans2.fit(synergy_coeff60_1[0:30:2])
# kmeans2.predict(synergy_coeff60_1[0:30:2])
# labels2 = kmeans2.labels_
#
# elbow_calculator(synergy_coeff90_1[0:30:2])
# kmeans3 = KMeans(n_clusters=3)
# kmeans3.fit(synergy_coeff90_1[0:30:2])
# kmeans3.predict(synergy_coeff90_1[0:30:2])
# labels3 = kmeans3.labels_
#
# p=np.vstack(((labels,labels2,labels3)))
#
# elbow_calculator(synergy_coeff0_1[1:30:2])
# kmeans = KMeans(n_clusters=2)
# kmeans.fit(synergy_coeff0_1[1:30:2])
# kmeans.predict(synergy_coeff0_1[1:30:2])
# labels = kmeans.labels_
#
# elbow_calculator(synergy_coeff20_1[1:30:2])
# kmeans1 = KMeans(n_clusters=2)
# kmeans1.fit(synergy_coeff20_1[1:30:2])
# kmeans1.predict(synergy_coeff20_1[1:30:2])
# labels1 = kmeans1.labels_
#
# elbow_calculator(synergy_coeff60_1[1:30:2])
# kmeans2 = KMeans(n_clusters=2)
# kmeans2.fit(synergy_coeff60_1[1:30:2])
# kmeans2.predict(synergy_coeff60_1[1:30:2])
# labels2 = kmeans2.labels_
#
# elbow_calculator(synergy_coeff90_1[1:30:2])
# kmeans3 = KMeans(n_clusters=2)
# kmeans3.fit(synergy_coeff90_1[1:30:2])
# kmeans3.predict(synergy_coeff90_1[1:30:2])
# labels3 = kmeans3.labels_
#
# p=np.vstack(((labels,labels2,labels3)))
#
#
# elbow_calculator(synergy_coeff20_1[0:30:2])
# kmeans = KMeans(n_clusters=2)
# kmeans.fit(synergy_coeff20_1[0:30:2])
# kmeans.predict(synergy_coeff20_1[0:30:2])
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
# plt.figure()
# plt.scatter(np.ravel(synergy_coeff20_1[0:30:2,0]),y=range(1,14),c=labels.astype(np.float),marker="o",  edgecolor='k')
# plt.scatter(np.ravel(synergy_coeff20_1[0:30:2,1]),y=range(1,14),c=labels.astype(np.float),marker="v", edgecolor='k')
# plt.scatter(np.ravel(synergy_coeff20_1[0:30:2,2]),y=range(1,14),c=labels.astype(np.float),marker="s", edgecolor='k')
# plt.scatter(np.ravel(synergy_coeff20_1[0:30:2,3]),y=range(1,14),c=labels.astype(np.float),marker="*", edgecolor='k')
# plt.scatter(np.ravel(synergy_coeff20_1[0:30:2,4]),y=range(1,14),c=labels.astype(np.float),marker="p", edgecolor='k')
#
# elbow_calculator(synergy_coeff60_1[0:30:2])
# kmeans = KMeans(n_clusters=2)
# kmeans.fit(synergy_coeff60_1[0:30:2])
# kmeans.predict(synergy_coeff60_1[0:30:2])
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
# plt.figure()
# plt.scatter(np.ravel(synergy_coeff60_1[0:30:2,0]),y=range(1,15),c=labels.astype(np.float),marker="o",  edgecolor='k')
# plt.scatter(np.ravel(synergy_coeff60_1[0:30:2,1]),y=range(1,15),c=labels.astype(np.float),marker="v", edgecolor='k')
# plt.scatter(np.ravel(synergy_coeff60_1[0:30:2,2]),y=range(1,15),c=labels.astype(np.float),marker="s", edgecolor='k')
# plt.scatter(np.ravel(synergy_coeff60_1[0:30:2,3]),y=range(1,15),c=labels.astyp0e(np.float),marker="*", edgecolor='k')
# plt.scatter(np.ravel(synergy_coeff60_1[0:30:2,4]),y=range(1,15),c=labels.astype(np.float),marker="p", edgecolor='k')
# labels = kmeans.predict(synergy_coeff0_1[0:30:2])

# a=np.asarray(range(0,14))
# a_1=np.asarray(range(0,13))
# b=np.asarray(list(itertools.combinations(a,2)))
# c=np.asarray(list(itertools.combinations(a_1,2)))
#
# print(synergy2_vector_correlation0_1.shape)
# print(b.shape)
#
# b[np.where(synergy2_vector_correlation0_1>0.6)]
# c[np.where(synergy2_vector_correlation20_1>0.6)]
# b[np.where(synergy2_vector_correlation60_1>0.6)]
# b[np.where(synergy2_vector_correlation90_1>0.6)]
#
# print(synergy_coeff0_1.shape)
#
average_synergy1_coeff0_1=np.zeros([1,5])
average_synergy2_coeff0_1=np.zeros([1,5])
for coeff in range(0,5):
    average_synergy1_coeff0_1[0,coeff]=np.mean(synergy_coeff0_1[0:30:2,coeff])
for coeff in range(0,5):
    average_synergy2_coeff0_1[0,coeff]=np.mean(synergy_coeff0_1[1:30:2,coeff])

average_synergy1_vector0_1=np.zeros([1,20000])
average_synergy2_vector0_1=np.zeros([1,20000])
for vector in range(0,20000):
    average_synergy1_vector0_1[0,vector]=np.mean(vector0_1[0:30:2,vector])
for vector in range(0,20000):
    average_synergy2_vector0_1[0,vector]=np.mean(vector0_1[1:30:2,vector])

average_synergy1_stuff0_1=np.zeros([1,20000])
average_synergy2_stuff0_1=np.zeros([1,20000])
average_synergy3_stuff0_1=np.zeros([1,20000])
average_synergy4_stuff0_1=np.zeros([1,20000])
average_synergy5_stuff0_1=np.zeros([1,20000])
average_stuff0_1=np.zeros([0,18000])
smooth = 1
for s in range(0,20000):
    average_synergy1_stuff0_1[0,s]=np.mean(stuff0_1[0:70:5,s])
# average_synergy1_stuff0_1 = amplitude_normalization(average_synergy1_stuff0_1)
average_synergy1_stuff0_1[0,:] = np.convolve(average_synergy1_stuff0_1[0,:], np.ones((smooth,))/smooth, mode='same')
average_stuff0_1 = np.append(average_stuff0_1, average_synergy1_stuff0_1[:,1000:19000], axis=0)
for s in range(0,20000):
    average_synergy2_stuff0_1[0,s]=np.mean(stuff0_1[1:70:5,s])
# average_synergy2_stuff0_1 = amplitude_normalization(average_synergy2_stuff0_1)
average_synergy2_stuff0_1[0,:] = np.convolve(average_synergy2_stuff0_1[0,:], np.ones((smooth,))/smooth, mode='same')
average_stuff0_1 = np.append(average_stuff0_1, average_synergy2_stuff0_1[:,1000:19000], axis=0)
for s in range(0,20000):
    average_synergy3_stuff0_1[0,s]=np.mean(stuff0_1[2:70:5,s])
# average_synergy3_stuff0_1 = amplitude_normalization(average_synergy3_stuff0_1)
average_synergy3_stuff0_1[0,:] = np.convolve(average_synergy3_stuff0_1[0,:], np.ones((smooth,))/smooth, mode='same')
average_stuff0_1 = np.append(average_stuff0_1, average_synergy3_stuff0_1[:,1000:19000], axis=0)
for s in range(0,20000):
    average_synergy4_stuff0_1[0,s]=np.mean(stuff0_1[3:70:5,s])
# average_synergy4_stuff0_1 = amplitude_normalization(average_synergy4_stuff0_1)
average_synergy4_stuff0_1[0,:] = np.convolve(average_synergy4_stuff0_1[0,:], np.ones((smooth,))/smooth, mode='same')
average_stuff0_1 = np.append(average_stuff0_1, average_synergy4_stuff0_1[:,1000:19000], axis=0)
for s in range(0,20000):
    average_synergy5_stuff0_1[0,s]=np.mean(stuff0_1[4:70:5,s])
# average_synergy5_stuff0_1 = amplitude_normalization(average_synergy5_stuff0_1)
average_synergy5_stuff0_1[0,:] = np.convolve(average_synergy5_stuff0_1[0,:], np.ones((smooth,))/smooth, mode='same')
average_stuff0_1 = np.append(average_stuff0_1, average_synergy5_stuff0_1[:,1000:19000], axis=0)



a,b,c=NMF_calculator(average_stuff0_1,rank=2)
# a,b,c=NMF_calculator(average_synergy1_stuff0_1,rank=2)
#
# average_synergy1_coeff20_1=np.zeros([1,5])
# average_synergy2_coeff20_1=np.zeros([1,5])
# for coeff in range(0,5):
#     average_synergy1_coeff20_1[0,coeff]=np.mean(synergy_coeff20_1[0:30:2,coeff])
# for coeff in range(0,5):
#     average_synergy2_coeff20_1[0,coeff]=np.mean(synergy_coeff20_1[1:30:2,coeff])
#
# average_synergy1_vector20_1=np.zeros([1,20000])
# average_synergy2_vector20_1=np.zeros([1,20000])
# for vector in range(0,20000):
#     average_synergy1_vector20_1[0,vector]=np.mean(vector20_1[0:30:2,vector])
# for vector in range(0,20000):
#     average_synergy2_vector20_1[0,vector]=np.mean(vector20_1[1:30:2,vector])
#
# average_synergy1_coeff60_1=np.zeros([1,5])
# average_synergy2_coeff60_1=np.zeros([1,5])
# for coeff in range(0,5):
#     average_synergy1_coeff60_1[0,coeff]=np.mean(synergy_coeff60_1[0:30:2,coeff])
# for coeff in range(0,5):
#     average_synergy2_coeff60_1[0,coeff]=np.mean(synergy_coeff60_1[1:30:2,coeff])
#0_1=np.zeros([1,5])
# average_synergy2_coeff20_1=np.zeros([1,5])
# for coeff in range(0,5):
#     average_synergy1_coeff20_1[0,coeff]=np.mean(synergy_coeff20_1[0:30:2,coeff])
# for coeff in range(0,5):
#     average_synergy2_coeff20_1[0,coeff]=np.mean(synergy_coeff20_1[1:30:2,coeff])
#
# average_synergy1_vector20_1=np.zeros([1,20000])
# average_synergy2_vector20_1=np.zeros([1,20000])
# for vector in range(0,20000):
#     average_synergy1_vector20_1[0,vector]=np.mean(vector20_1[0:30:2,vector])
# for vector in range(0,20000):
#     average_synergy2_vector20_1[0,vector]=np.mean(vector20_1[1:30:2,vector])
# average_synergy1_vector60_1=np.zeros([1,20000])
# average_synergy2_vector60_1=np.zeros([1,20000])
# for vector in range(0,20000):
#     average_synergy1_vector60_1[0,vector]=np.mean(vector60_1[0:30:2,vector])
# for vector in range(0,20000):
#     average_synergy2_vector60_1[0,vector]=np.mean(vector60_1[1:30:2,vector])

# average_synergy1_coeff90_1=np.zeros([1,5])
# average_synergy2_coeff90_1=np.zeros([1,5])
# for coeff in range(0,5):
#     average_synergy1_coeff90_1[0,coeff]=np.mean(synergy_coeff90_1[0:30:2,coeff])
# for coeff in range(0,5):
#     average_synergy2_coeff90_1[0,coeff]=np.mean(synergy_coeff90_1[1:30:2,coeff])
#
# average_synergy1_vector90_1=np.zeros([1,20000])
# average_synergy2_vector90_1=np.zeros([1,20000])
# for vector in range(0,20000):
#     average_synergy1_vector90_1[0,vector]=np.mean(vector90_1[0:30:2,vector])
# for vector in range(0,20000):
#     average_synergy2_vector90_1[0,vector]=np.mean(vector90_1[1:30:2,vector])
#
# synergy1_coeff_comparison_0to20=pairwise.cosine_similarity(average_synergy1_coeff0_1,average_synergy1_coeff20_1)
# synergy1_coeff_comparison_0to60=pairwise.cosine_similarity(average_synergy1_coeff0_1,average_synergy1_coeff60_1)
# synergy1_coeff_comparison_0to90=pairwise.cosine_similarity(average_synergy1_coeff0_1,average_synergy1_coeff90_1)
# synergy1_coeff_comparison_20to60=pairwise.cosine_similarity(average_synergy1_coeff20_1,average_synergy1_coeff60_1)
# synergy1_coeff_comparison_20to90=pairwise.cosine_similarity(average_synergy1_coeff20_1,average_synergy1_coeff90_1)
# synergy1_coeff_comparison_60to90=pairwise.cosine_similarity(average_synergy1_coeff60_1,average_synergy1_coeff90_1)
#
# synergy1_vector_comparison_0to20=pairwise.cosine_similarity(average_synergy1_vector0_1,average_synergy1_vector20_1)
# synergy1_vector_comparison_0to60=pairwise.cosine_similarity(average_synergy1_vector0_1,average_synergy1_vector60_1)
# synergy1_vector_comparison_0to90=pairwise.cosine_similarity(average_synergy1_vector0_1,average_synergy1_vector90_1)
# synergy1_vector_comparison_20to60=pairwise.cosine_similarity(average_synergy1_vector20_1,average_synergy1_vector60_1)
# synergy1_vector_comparison_20to90=pairwise.cosine_similarity(average_synergy1_vector20_1,average_synergy1_vector90_1)
# synergy1_vector_comparison_60to90=pairwise.cosine_similarity(average_synergy1_vector60_1,average_synergy1_vector90_1)
#
# synergy2_coeff_comparison_0to20=pairwise.cosine_similarity(average_synergy2_coeff0_1,average_synergy2_coeff20_1)
# synergy2_coeff_comparison_0to60=pairwise.cosine_similarity(average_synergy2_coeff0_1,average_synergy2_coeff60_1)
# synergy2_coeff_comparison_0to90=pairwise.cosine_similarity(average_synergy2_coeff0_1,average_synergy2_coeff90_1)
# synergy2_coeff_comparison_20to60=pairwise.cosine_similarity(average_synergy2_coeff20_1,average_synergy2_coeff60_1)
# synergy2_coeff_comparison_20to90=pairwise.cosine_similarity(average_synergy2_coeff20_1,average_synergy2_coeff90_1)
# synergy2_coeff_comparison_60to90=pairwise.cosine_similarity(average_synergy2_coeff60_1,average_synergy2_coeff90_1)
#
# synergy2_vector_comparison_0to20=pairwise.cosine_similarity(average_synergy2_vector0_1,average_synergy2_vector20_1)
# synergy2_vector_comparison_0to60=pairwise.cosine_similarity(average_synergy2_vector0_1,average_synergy2_vector60_1)
# synergy2_vector_comparison_0to90=pairwise.cosine_similarity(average_synergy2_vector0_1,average_synergy2_vector90_1)
# synergy2_vector_comparison_20to60=pairwise.cosine_similarity(average_synergy2_vector20_1,average_synergy2_vector60_1)
# synergy2_vector_comparison_20to90=pairwise.cosine_similarity(average_synergy2_vector20_1,average_synergy2_vector90_1)
# synergy2_vector_comparison_60to90=pairwise.cosine_similarity(average_synergy2_vector60_1,average_synergy2_vector90_1)

index = np.arange(5)
names=("RF","VL","VM","ST","BF")
percentage = ("0","25","50","75","100")

def plot_line(x):
    plt.figure(1)
    plt.xticks(index,names)
    plt.ylim(0,1)
    plt.yticks(np.linspace(0,1,num=5),percentage)
    plt.hlines(y=0.75,xmin=0,xmax=4,linestyles='dashed')
    for line in range(0,np.shape(x)[0]):
        plt.plot(index,x[line,:])

# plot_line(vaf_muscle0_1)
# plot_line(vaf_muscle20_1)
# plot_line(vaf_muscle60_1)
# plot_line(vaf_muscle90_1)

plt.figure()
plt.subplot(211)
plt.plot(average_synergy1_vector0_1[0])
plt.subplot(212)
plt.plot(average_synergy2_vector0_1[0])
plt.show()
# plt.plot(average_synergy1_vector20_1[0])
# plt.show()
# plt.plot(average_synergy2_vector20_1[0])
# plt.show()
# plt.plot(average_synergy1_vector60_1[0])
# plt.plot(average_synergy2_vector60_1[0])
# plt.plot(average_synergy1_vector90_1[0])
# plt.show()
# plt.plot(average_synergy2_vector90_1[0])
# plt.show()

plt.figure()
plt.subplot(511)
plt.plot(average_stuff0_1[0])
plt.subplot(512)
plt.plot(average_stuff0_1[1])
plt.subplot(513)
plt.plot(average_stuff0_1[2])
plt.subplot(514)
plt.plot(average_stuff0_1[3])
plt.plot(average_stuff0_1[0])
plt.subplot(515)
plt.plot(average_stuff0_1[4])
plt.plot(average_stuff0_1[0])
plt.figure()
plt.subplot(411)
a = np.transpose(a)
plt.bar(index,np.ravel(a[0]),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))
plt.subplot(412)
plt.bar(index,np.ravel(a[1]),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))
plt.subplot(413)
plt.plot(c[0].tolist()[0])
plt.subplot(414)
plt.plot(c[1].tolist()[0])
# plt.plot(vector20_1[0])
# plt.plot(vector20_1[1])
# plt.plot(vector20_1[2])
# plt.plot(vector20_1[3])
# plt.plot(vector20_1[4])
# plt.plot(vector20_1[3])
# plt.plot(vector60_1[2])
# plt.plot(vector60_1[3])
# plt.plot(vector90_1[0])
# plt.plot(vector90_1[1])
# plt.plot(vector90_1[2])
# plt.plot(vector90_1[3])
# plt.plot(vector90_1[4])
plt.show()
# plt.plot(vector0_1[4])
# plt.plot(vector0_1[5])
# plt.plot(vector0_1[6])
# plt.plot(vector0_1[7])
# plt.plot(vector0_1[8])
# plt.plot(vector0_1[9])
# plt.plot(vector0_1[10])
# plt.plot(vector0_1[11])
# plt.plot(vector0_1[12])
# plt.plot(vector0_1[13])
# plt.plot(vector0_1[14])
# plt.plot(vector0_1[15])
# plt.plot(vector0_1[16])
# plt.plot(vector0_1[17])
# plt.plot(vector0_1[18])
# plt.plot(vector0_1[19])
# plt.plot(vector0_1[20])
# plt.plot(vector0_1[21])
# plt.plot(vector0_1[22])
# plt.plot(vector0_1[23])
# plt.plot(vector0_1[24])
# plt.plot(vector0_1[25])
# plt.plot(vector0_1[26])
# plt.plot(vector0_1[27])


plt.figure()
plt.bar(index,np.ravel(average_synergy1_coeff0_1),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))
# plt.bar(index,np.ravel(average_synergy1_coeff20_1),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))
# plt.bar(index,np.ravel(average_synergy1_coeff60_1),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))
# plt.bar(index,np.ravel(average_synergy1_coeff90_1),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))

plt.show()

plt.figure()
plt.bar(index,np.ravel(average_synergy2_coeff0_1),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))
# plt.bar(index,np.ravel(average_synergy2_coeff20_1),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))
# plt.bar(index,np.ravel(average_synergy2_coeff60_1),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))
# plt.bar(index,np.ravel(average_synergy2_coeff90_1),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))

# plt.figure()
# plt.bar(index,np.ravel(average_synergy3_coeff0_1),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))
# plt.bar(index,np.ravel(average_synergy2_coeff20_1),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))
# plt.bar(index,np.ravel(average_synergy2_coeff60_1),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))
# plt.bar(index,np.ravel(average_synergy2_coeff90_1),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))

plt.show()
