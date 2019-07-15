
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
from sklearn.decomposition import NMF

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
    nmf = nimfa.Nmf(abs(x), seed='nndsvd', rank=rank, max_iter=50)
    nmf_fit = nmf()
    print('Evar: %5.4f' % nmf_fit.fit.evar())
    vector=np.asarray(nmf_fit.basis())
    synergy_coeff=nmf_fit.fit.coef()
    residuals=np.asarray(nmf_fit.fit.residuals())

    d = nmf.estimate_rank(rank_range=[1,2,3,4,5], n_run=10, idx=0, what=['residuals'])
    rss = []
    r_i = 0
    vaf_muscle=np.zeros([1,x.shape[1]])
    for r, dd in d.items():
        res = dd['residuals']
        # print(res)
        # result = (res**2)/(np.sum(np.square(x - np.mean(x))))
        # print('blah')
        # print(result)
        # rss = rss + [result]

        avg_res = 0
        print('before')
        print(res)
        for column in range(0,x.shape[1]):
            result = 1-np.abs(np.sum(np.square(res))/(np.sum(np.square(x[:,column]))))
            avg_res = avg_res + result
            # vaf_muscle[r_i,column] = result

        r_i = r_i + 1
        avg_res = avg_res / x.shape[1]
        rss = rss + [avg_res]

    print('RSS WITH VAR')
    print(rss)
    return vector,vaf_muscle,synergy_coeff,rss

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
    nmf = nimfa.Nmf(abs(x), seed=None, W=start_vector, H=start_coeff, rank=rank, max_iter=500)
    nmf_other = NMF(solver='mu', n_components=rank, init='nndsvd')
    W = nmf_other.fit_transform(x)
    H = nmf_other.components_
    nmf_fit = nmf()
    print('Evar: %5.4f' % nmf_fit.fit.evar())
    vector=np.asarray(nmf_fit.basis())
    synergy_coeff=nmf_fit.fit.coef()
    residuals=np.asarray(nmf_fit.fit.residuals())
    vaf_muscle=np.zeros([1,x.shape[1]])
    for column in range(0,x.shape[1]):
        result = 1-(sum(residuals[:,column]**2)/sum(x[:,column]**2))
        vaf_muscle[0,column] = result
    return W,vaf_muscle,H

def synergy_analysis_shared(directory):

    directorys=["S1_seperated by angle/p1","S1_seperated by angle/p2","S1_seperated by angle/p3","S1_seperated by angle/p4","S1_seperated by angle/p5","S1_seperated by angle/p6","S1_seperated by angle/p7","S1_seperated by angle/p8","S1_seperated by angle/p9","S1_seperated by angle/p10","S1_seperated by angle/p11","S1_seperated by angle/p12","S1_seperated by angle/p13","S1_seperated by angle/p14"]
    time_series_length = 20000
    t_start = 18000
    t_end = 38000
    num_angles = 4
    num_coefficients = 5
    rank = 2
    times = [20000,49000,81000,112000,141000]
    # times = [20000]

    datafile_avg = np.zeros([time_series_length,num_coefficients])
    for d in directorys:
        datafile_action_avg = np.zeros([time_series_length,num_coefficients])
        for t in times:
            files=[]
            for file in os.listdir(d):
                if file.endswith(".csv"):
                    files.append(file)
            datafile_full = np.zeros([0,num_coefficients])
            for file in files:
                if "_0deg" not in file:
                    continue
                wholefile=np.genfromtxt(os.path.join(d, file),delimiter=',',skip_header=1)
                df1=datafile_generator(wholefile)
                datafile1 = df1[t-2000:t+18000]
                for column in range(0,datafile1.shape[1]):
                    datafile1[:,column]=butter_bandpass_filter(datafile1[:,column],lowcut=20,highcut=450,fs=2000,order=2)
                    datafile1[:,column]=butter_lowpass_filter(abs(datafile1[:,column]),lowcut=20,fs=2000,order=2)
                    datafile1[:,column] = np.convolve(datafile1[:,column], np.ones((500,))/500, mode='same')
                    datafile1[:,column]=amplitude_normalization(datafile1[:,column])

                datafile_full = np.append(datafile_full, datafile1, axis=0)
            datafile_action_avg += datafile_full
        datafile_action_avg = datafile_action_avg / len(times)
        datafile_avg += datafile_action_avg

    datafile_avg = datafile_avg / len(directorys)

    # for column in range(0,datafile_avg.shape[1]):
    #     datafile_avg[:,column]=amplitude_normalization(datafile_avg[:,column])

    start_coeff = np.random.rand(rank,num_coefficients)
    build_zeros = np.zeros([time_series_length,rank])
    build_random = np.random.rand(time_series_length,rank)

    print(start_coeff.shape)

    top_row = build_random
    # top_row = np.append(top_row, build_random, axis=0)
    # top_row = np.append(top_row, build_random, axis=0)
    # top_row = np.append(top_row, build_random, axis=0)

    print(top_row.shape)

    a,b,c=NMF_calculator_shared(datafile_avg,top_row, start_coeff, rank=rank)

    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(np.ravel(c[0].tolist()))
    print(np.ravel(c[1].tolist()))

    index = np.arange(num_coefficients)
    plt.figure()

    a = np.transpose(a)
    datafile_avg=np.transpose(datafile_avg)
    plt.subplot(411)
    plt.bar(index,np.ravel(c[0].tolist()),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))
    plt.subplot(412)
    plt.bar(index,np.ravel(c[1].tolist()),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))
    plt.subplot(413)
    plt.plot(a[0].tolist())
    plt.subplot(414)
    plt.plot(a[1].tolist())
    plt.show()
    return a,c,datafile_avg

def synergy_analysis(directory):
    vector = np.zeros([20000,0])
    stuff = np.zeros([20000,0])
    vaf_muscle = np.zeros([0,5])
    rss_nums = np.zeros([0,5])
    synergy_coeff = np.zeros([0,5])
    synergy1_vector_correlation = []
    synergy2_vector_correlation = []
    synergy3_vector_correlation = []
    synergy1_coeff_correlation = []
    synergy2_coeff_correlation = []
    synergy3_coeff_correlation = []

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
                datafile1[:,column] = np.convolve(datafile1[:,column], np.ones((500,))/500, mode='same')
                datafile1[:,column]=amplitude_normalization(datafile1[:,column])

                # plt.subplot(511 + column)
                # plt.plot(datafile1[:,column].tolist())

                # datafile1[0:1000,column] = np.zeros(1000)
                # datafile1[200000:20000,column] = np.zeros(1000)
            stuff = np.append(stuff, datafile1, axis=1)
            print(stuff.shape)
            a,b,c,rss=NMF_calculator(datafile1,rank=2)

            print("abc:")
            print(a.shape)
            print(b.shape)
            print(c.shape)
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
            print(np.array(rss).shape)
            rs = np.reshape(np.array(rss), [1,5])
            rss_nums=np.append(rss_nums,rs,axis=0)
            print(vaf_muscle.shape)
            synergy_coeff=np.append(synergy_coeff,c,axis=0)
            print(synergy_coeff.shape)
    vector=np.transpose(vector)
    stuff=np.transpose(stuff)
    print('******* VAF *******')
    print(rss_nums)
    print(np.mean(rss_nums, axis = 0))
    # plt.figure()
    # plt.plot(np.mean(rss_nums, axis = 0))
    # plt.show()
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
    return vector,vaf_muscle,synergy_coeff,synergy1_vector_correlation,synergy2_vector_correlation,synergy1_coeff_correlation,synergy2_coeff_correlation,stuff,rss_nums

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
for muscle in range(0,5):figure()
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
rss_n = np.zeros([0,5])
# vaf_m = np.zeros([0,5])
# vector0_1,vaf_muscle0_1,synergy_coeff0_1,synergy1_vector_correlation0_1,synergy2_vector_correlation0_1,synergy1_coeff_correlation0_1,synergy2_coeff_correlation0_1,stuff0_1,rss_nums=synergy_analysis(directory=directory0_1)
# print(vaf_muscle0_1.shape)
# print(vaf_muscle0_1)
# rss_n=np.append(rss_n,rss_nums,axis=0)
# vaf_m=np.append(vaf_m,vaf_muscle0_1,axis=0)
# vector0_1,vaf_muscle0_1,synergy_coeff0_1,synergy1_vector_correlation0_1,synergy2_vector_correlation0_1,synergy1_coeff_correlation0_1,synergy2_coeff_correlation0_1,stuff0_1,rss_nums=synergy_analysis(directory=directory20_1)
# rss_n=np.append(rss_n,rss_nums,axis=0)
# vaf_m=np.append(vaf_m,vaf_muscle0_1,axis=0)
# vector0_1,vaf_muscle0_1,synergy_coeff0_1,synergy1_vector_correlation0_1,synergy2_vector_correlation0_1,synergy1_coeff_correlation0_1,synergy2_coeff_correlation0_1,stuff0_1,rss_nums=synergy_analysis(directory=directory60_1)
# rss_n=np.append(rss_n,rss_nums,axis=0)
# vaf_m=np.append(vaf_m,vaf_muscle0_1,axis=0)
vector0_1,vaf_muscle0_1,synergy_coeff0_1,synergy1_vector_correlation0_1,synergy2_vector_correlation0_1,synergy1_coeff_correlation0_1,synergy2_coeff_correlation0_1,stuff0_1,rss_nums=synergy_analysis(directory=directory90_1)
rss_n=np.append(rss_n,rss_nums,axis=0)
# vaf_m=np.append(vaf_m,vaf_muscle0_1,axis=0)
print(np.mean(rss_n, axis = 0))
# print(np.mean(vaf_m, axis = 0))
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=16)
# plt.bar([1,2,3,4,5],np.mean(rss_n, axis = 0)*100,tick_label=("1","2","3","4","5"), color='#999999', edgecolor='k', linewidth='1')
ax.errorbar([1,2,3,4,5],np.mean(rss_n, axis = 0),yerr=stats.sem(rss_n, axis = 0),color='#000000', fmt='-')
ax.plot([1,2,3,4,5],[0.9,0.9,0.9,0.9,0.9], '--', color='#888888')
plt.xlabel('NMF Rank', fontsize=20)
plt.ylabel('VAF', fontsize=20)
plt.ylim(bottom=0.5, top=1)
plt.show()
names=("RF","VL","VM","ST","BF")

# vector0_1,synergy_coeff0_1,stuff0_1 = synergy_analysis_shared(directory="S1_seperated by angle/p2")

average_synergy1_coeff0_1=np.zeros([1,5])
average_synergy2_coeff0_1=np.zeros([1,5])
# average_synergy3_coeff0_1=np.zeros([1,5])
for coeff in range(0,5):
    average_synergy1_coeff0_1[0,coeff]=np.mean(synergy_coeff0_1[0:30:2,coeff])
for coeff in range(0,5):
    average_synergy2_coeff0_1[0,coeff]=np.mean(synergy_coeff0_1[1:30:2,coeff])
# for coeff in range(0,5):
#     average_synergy3_coeff0_1[0,coeff]=np.mean(synergy_coeff0_1[2:30:3,coeff])

average_synergy1_vector0_1=np.zeros([1,20000])
average_synergy2_vector0_1=np.zeros([1,20000])
# average_synergy3_vector0_1=np.zeros([1,20000])
for vector in range(0,20000):
    average_synergy1_vector0_1[0,vector]=np.mean(vector0_1[0:30:2,vector])
for vector in range(0,20000):
    average_synergy2_vector0_1[0,vector]=np.mean(vector0_1[1:30:2,vector])
# for vector in range(0,20000):
#     average_synergy3_vector0_1[0,vector]=np.mean(vector0_1[2:30:3,vector])

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



a,b,c,rss=NMF_calculator(average_stuff0_1,rank=2)

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

plt.figure()
plt.subplot(211)
plt.plot(average_synergy1_vector0_1[0])
plt.subplot(212)
plt.plot(average_synergy2_vector0_1[0])
# plt.subplot(313)
# plt.plot(average_synergy3_vector0_1[0])
plt.show()

np.savetxt('average_raw.csv', np.transpose(average_stuff0_1), delimiter=",")

plt.figure()
plt.subplot(511)
plt.plot(average_stuff0_1[0])
plt.subplot(512)
plt.plot(average_stuff0_1[1])
plt.subplot(513)
plt.plot(average_stuff0_1[2])
plt.subplot(514)
plt.plot(average_stuff0_1[3])
plt.subplot(515)
plt.plot(average_stuff0_1[4])
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

plt.show()


plt.figure()
plt.bar(index,np.ravel(average_synergy1_coeff0_1),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))

plt.figure()
plt.bar(index,np.ravel(average_synergy2_coeff0_1),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))

plt.show()
