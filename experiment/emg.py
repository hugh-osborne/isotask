
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
    # print('Evar: %5.4f' % nmf_fit.fit.evar())
    vector=np.asarray(nmf_fit.basis())
    synergy_coeff=nmf_fit.fit.coef()

    vaf_muscle=np.zeros([1,x.shape[1]])

    rss = []
    return vector,vaf_muscle,synergy_coeff,rss

def NMF_calculator_with_VAF_check(x, rank):
    nmf = nimfa.Nmf(abs(x), seed='nndsvd', rank=rank, max_iter=50)
    nmf_fit = nmf()
    #print('Evar: %5.4f' % nmf_fit.fit.evar())
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
        #print('before')
        #print(res)
        for column in range(0,x.shape[1]):
            result = 1-np.abs(np.sum(np.square(res))/(np.sum(np.square(x[:,column]))))
            avg_res = avg_res + result
            # vaf_muscle[r_i,column] = result

        r_i = r_i + 1
        avg_res = avg_res / x.shape[1]
        rss = rss + [avg_res]

    #print('RSS WITH VAR')
    #print(rss)
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

def synergy_analysis(folder, num_files, degrees, ax1, ax2, pos1, synergy):

    directorys=[folder + "/p1",folder + "/p2",folder + "/p3",folder + "/p4",folder + "/p5",folder + "/p6",folder + "/p7",folder + "/p8",folder + "/p9",folder + "/p10",folder + "/p11",folder + "/p12",folder + "/p13",folder + "/p14"]
    directorys=directorys[0:num_files]
    time_series_length = 20000
    time_series_offset = 2000
    num_coefficients = 5
    rank = 2
    times = [20000,49000,81000,112000,141000] #
    blur_window = 1000

    start_coeff = np.random.rand(rank,num_coefficients)
    build_random = np.random.rand(time_series_length,rank)
    top_row = build_random

    datafile_avg = np.zeros([time_series_length,num_coefficients])
    a_avg = np.zeros([time_series_length-(blur_window*2),rank])
    c_avg = np.zeros([rank, num_coefficients])
    a_s = np.zeros([time_series_length-(blur_window*2),rank])
    c_s = np.zeros([rank, num_coefficients])
    missing_file_count = 0
    for d in directorys:
        datafile_action_avg = np.zeros([time_series_length,num_coefficients])
        a_action_avg = np.zeros([time_series_length-(blur_window*2),rank])
        c_action_avg = np.zeros([rank, num_coefficients])
        for t in times:
            files=[]
            for file in os.listdir(d):
                if file.endswith(".csv"):
                    files.append(file)
            datafile_full = np.zeros([0,num_coefficients])
            has_degrees = False
            for file in files:
                s = '_' + str(degrees) + 'deg'
                if s not in file:
                    continue
                has_degrees = True
                wholefile=np.genfromtxt(os.path.join(d, file),delimiter=',',skip_header=1)
                df1=datafile_generator(wholefile)
                datafile1 = df1[t-time_series_offset:t+(time_series_length-time_series_offset)]
                for column in range(0,datafile1.shape[1]):
                    datafile1[:,column]=butter_bandpass_filter(datafile1[:,column],lowcut=20,highcut=450,fs=2000,order=2)
                    datafile1[:,column]=butter_lowpass_filter(abs(datafile1[:,column]),lowcut=20,fs=2000,order=2)
                    datafile1[:,column] = np.convolve(datafile1[:,column], np.ones((blur_window,))/blur_window, mode='same')
                    datafile1[:,column]=amplitude_normalization(datafile1[:,column])

                a,b,c,rss=NMF_calculator(datafile1[blur_window:time_series_length-blur_window],rank=rank)
                a_action_avg += a
                c_action_avg += c
                print("********************************************************************")
                print("********************************************************************")
                print("********************************************************************")
                
                print(np.ravel(c[0]), np.mean(np.ravel(c[0])), np.std(np.ravel(c[0])), stats.iqr(np.ravel(c[0])))
                
                print("********************************************************************")
                print("********************************************************************")
                print("********************************************************************")
                # if len([cc for cc in np.ravel(c[0]) if cc > 5 and cc < 6.3]) == 5:
                a_s = np.append(a_s,a,axis=1)
                c_s = np.append(c_s,c,axis=0)
                # print(d, t)
                # index = np.arange(num_coefficients)
                # col='red'
                # plt.figure()
                # plt.subplot(411)
                # plt.bar(index,np.ravel(c[0].tolist()), 0.4, color=col, alpha=1.0, edgecolor='#000000',linewidth=3, hatch="//")
                # plt.subplot(412)
                # plt.bar(index,np.ravel(c[1].tolist()), 0.4,  color=col,alpha=1.0, edgecolor='#000000',linewidth=3, hatch="//")
                # # plt.subplot(613)
                # # plt.bar(index,np.ravel(c[2].tolist()),tick_label=("RF","VL","VM","ST","BF"),color=("red","blue","green","gold","deeppink"))
                # plt.subplot(413)
                # plt.plot(a[:,0].tolist())
                # plt.subplot(414)
                # plt.plot(a[:,1].tolist())
                # # plt.subplot(616)
                # # plt.plot(a[2].tolist())
                # plt.show()

                datafile_full = np.append(datafile_full, datafile1, axis=0)
            if has_degrees:
                datafile_action_avg += datafile_full
            else:
                missing_file_count += 1
                continue
        datafile_action_avg = datafile_action_avg / len(times)
        a_action_avg = a_action_avg / len(times)
        c_action_avg = c_action_avg / len(times)
        datafile_avg += datafile_action_avg
        a_avg += a_action_avg
        c_avg += c_action_avg

    datafile_avg = datafile_avg / (len(directorys)-missing_file_count)
    a_avg = a_avg / (len(directorys)-missing_file_count)
    c_avg = c_avg / (len(directorys)-missing_file_count)


    a_avg[:,0] = np.mean(a_s[:,2::2], axis=1)
    a_avg[:,1] = np.mean(a_s[:,3::2], axis=1)

    c_avg[0] = np.mean(c_s[2::2], axis=0)
    c_avg[1] = np.mean(c_s[3::2], axis=0)

    show_muscle_labels=True
    col='red'
    if not pos1:
        col='#222288'

    a_avg = np.transpose(a_avg)
    e = stats.sem(c_s[2::2], axis=0)

    if synergy == 1:

        ax1.set_ylim([0,6.9])
        index_only = np.array(range(0,5))
        index = np.array(range(0,5)) - np.ones(5)*0.2 
        if not pos1:
            index = np.array(range(0,5)) + np.ones(5)*0.2 
        ax1.set_xticks(index_only)
        ax1.set_xticklabels(['RF','VL','VM','ST','BF'])
        if pos1:
            rects1 = ax1.bar(index,np.ravel(c_avg[0].tolist()), 0.4, yerr=np.ravel(e), color=col, error_kw=dict(lw=3, capsize=4, capthick=3), label='Position 1',capsize=4, alpha=1.0, edgecolor='#000000',linewidth=3, hatch="//")
        else:
            rects1 = ax1.bar(index,np.ravel(c_avg[0].tolist()), 0.4, yerr=np.ravel(e), color=col, error_kw=dict(lw=3, capsize=4, capthick=3), label='Position 1',capsize=4, alpha=1.0, edgecolor='#000000',linewidth=3)
        rects1[0].set_color(col)
        rects1[1].set_color(col)
        rects1[2].set_color(col)
        rects1[3].set_color(col)
        rects1[4].set_color(col)
        rects1[0].set_edgecolor('#000000')
        rects1[1].set_edgecolor('#000000')
        rects1[2].set_edgecolor('#000000')
        rects1[3].set_edgecolor('#000000')
        rects1[4].set_edgecolor('#000000')
        ax1.tick_params(axis='both',which='both',left=False,bottom=False,labelbottom=show_muscle_labels,labelleft=False,labelsize=28)
        ax1.yaxis.set_ticks(np.arange(0.0, 6.91, 1.0))
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.plot()

        ax2.set_ylim([0,0.15])
        rects1 = ax2.plot(a_avg[0].tolist(), color=col, linewidth=3)
        ax2.tick_params(axis='both',which='both',left=False,bottom=False,labelbottom=False,labelleft=False,labelsize=28)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.plot()

    if synergy == 2:

        e = stats.sem(c_s[3::2], axis=0)

        ax1.set_ylim([0,4.8])
        index_only = np.array(range(0,5))
        index = np.array(range(0,5)) - np.ones(5)*0.2 
        if not pos1:
            index = np.array(range(0,5)) + np.ones(5)*0.2 
        ax1.set_xticks(index_only)
        ax1.set_xticklabels(['RF','VL','VM','ST','BF'])
        if pos1:
            rects1 = ax1.bar(index,np.ravel(c_avg[1].tolist()), 0.4, yerr=np.ravel(e), color=col, error_kw=dict(lw=3, capsize=4, capthick=3), label='Position 1',capsize=4, alpha=1.0, edgecolor='#000000',linewidth=3, hatch="//")
        else:
            rects1 = ax1.bar(index,np.ravel(c_avg[1].tolist()), 0.4, yerr=np.ravel(e), color=col, error_kw=dict(lw=3, capsize=4, capthick=3), label='Position 1',capsize=4, alpha=1.0, edgecolor='#000000',linewidth=3, hatch="//")
        rects1[0].set_color(col)
        rects1[1].set_color(col)
        rects1[2].set_color(col)
        rects1[3].set_color(col)
        rects1[4].set_color(col)
        rects1[0].set_edgecolor('#000000')
        rects1[1].set_edgecolor('#000000')
        rects1[2].set_edgecolor('#000000')
        rects1[3].set_edgecolor('#000000')
        rects1[4].set_edgecolor('#000000')
        ax1.tick_params(axis='both',which='both',left=False,bottom=False,labelbottom=show_muscle_labels,labelleft=False,labelsize=28)
        ax1.yaxis.set_ticks(np.arange(0.0, 4.81, 1.0))
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.plot()

        ax2.set_ylim([0,0.06])
        rects1 = ax2.plot(a_avg[1].tolist(), color=col, linewidth=3)
        ax2.tick_params(axis='both',which='both',left=False,bottom=False,labelbottom=False,labelleft=False,labelsize=28)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.plot()

    return a,c,datafile_avg

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

# Check raw EMG data
def check_smoothed_raw(position, angle, normalise=False):
    if position == 1:
        folder = "Position_Participant_Angle/P1_by_participant"
        num_files = 10
        directorys=[folder + "/p1",folder + "/p2",folder + "/p4",folder + "/p5",folder + "/p6",folder + "/p8",folder + "/p9"]
    elif position == 2:
        folder = "Position_Participant_Angle/P2_by_participant"
        num_files = 8
        directorys=[folder + "/p2",folder + "/p4",folder + "/p5",folder + "/p6",folder + "/p7",folder + "/p9",folder + "/p10",folder + "/p11"]
    else:
        return
    
    num_coefficients = 7
    
    fig, axes = plt.subplots(len(directorys),num_coefficients,figsize=(8,8))
    fig.subplots_adjust(top=0.95)
    fig.set_size_inches(18, 18)
        
    for d in range(len(directorys)):
        print("Processing Directory: ", directorys[d])
        files=[]
        for file in os.listdir(directorys[d]):
            if file.endswith(".csv"):
                files.append(file)
                
        for file in range(len(files)):
            s = '_' + str(angle) + 'deg'
            if s not in files[file]:
                continue
            print("File: ", files[file])
            wholefile=np.genfromtxt(os.path.join(directorys[d], files[file]),delimiter=',',skip_header=1)
            df1=datafile_generator(wholefile)
                      
            for column in range(0,df1.shape[1]):
                df1[:,column]=butter_lowpass_filter(abs(df1[:,column]),lowcut=20,fs=20000,order=2)
                if normalise:
                   df1[:,column]=amplitude_normalization(df1[:,column])
                axes[d][column].plot(df1[:,column].tolist(), color='#222288', linewidth=3)
                axes[d][column].set_ylim([0,0.00006])
    plt.show()
                
                

# Get smoothed and normalised EMG output
def get_smoothed_emg_output(position, angle, normalise=True):
    if position == 1:
        folder = "Position_Participant_Angle/P1_by_participant"
        num_files = 14
        directorys=[folder + "/p1",folder + "/p2",folder + "/p4",folder + "/p5",folder + "/p6",folder + "/p8",folder + "/p9"]
    elif position == 2:
        folder = "Position_Participant_Angle/P2_by_participant"
        num_files = 11
        directorys=[folder + "/p2",folder + "/p4",folder + "/p5",folder + "/p6",folder + "/p7",folder + "/p9",folder + "/p10",folder + "/p11"]
    else:
        return

    directorys=[folder + "/p1",folder + "/p2",folder + "/p3",folder + "/p4",folder + "/p5",folder + "/p6",folder + "/p7",folder + "/p8",folder + "/p9",folder + "/p10",folder + "/p11",folder + "/p12",folder + "/p13",folder + "/p14"]
    directorys=directorys[0:num_files]
    #directorys=[folder + "/p1",folder + "/p2"]
    time_series_length = 20000
    time_series_offset = 2000
    time_margin = 0
    time_series_length = time_series_length - (2*time_margin)
    num_coefficients = 5
    degrees = angle
    times = [20000,49000,81000,112000,141000]#,49000,81000,112000,141000] #
    
    datafile_avg = np.empty([0,time_series_length,num_coefficients]) #[np.zeros([time_series_length,num_coefficients])]
    missing_file_count = 0
    for d in directorys:
        print("Processing Directory: ", d)
        datafile_action_avg = np.empty([0,time_series_length, num_coefficients])#np.zeros([time_series_length,num_coefficients])
        files=[]
        for file in os.listdir(d):
            if file.endswith(".csv"):
                files.append(file)
        has_degrees = False
        for file in files:
            s = '_' + str(degrees) + 'deg'
            if s not in file:
                continue
            print("File: ", file)
            has_degrees = True
            datafile_full = []#np.empty([time_series_length,num_coefficients]) #np.zeros([0,num_coefficients])
            
            wholefile=np.genfromtxt(os.path.join(d, file),delimiter=',',skip_header=1)
            df1=datafile_generator(wholefile)
                
            for t in times:
                datafile1 = df1[t-time_series_offset+time_margin:t+(time_series_length-time_series_offset)+time_margin]
                if (len(datafile1) < time_series_length):
                    continue
                    
                for column in range(0,datafile1.shape[1]):
                    datafile1[:,column]=butter_lowpass_filter(1000*abs(datafile1[:,column]),lowcut=20,fs=40000,order=2)
                    #if normalise:
                    #    datafile1[:,column]=amplitude_normalization(datafile1[:,column])
                    #diff = max(datafile1[:,column]) - min(datafile1[:,column])
                    #mmin = min(datafile1[:,column])
                    #m = np.mean([a for a in datafile1[:,column] if a < mmin + (diff/2.0)])
                    
                    #datafile1[:,column] = [a - mmin for a in datafile1[:,column]]
                    
                    if normalise:
                        datafile1[:,column]=amplitude_normalization(datafile1[:,column])
                    
                datafile_full = datafile_full + [datafile1]
            if has_degrees:
                datafile_action_avg = np.append(datafile_action_avg,datafile_full,axis=0)
            else:
                missing_file_count += 1
                continue
        datafile_avg = np.append(datafile_avg, datafile_action_avg, axis=0)
        
    #datafile_avg = datafile_avg / (len(directorys)-missing_file_count)
    
    return datafile_avg
    
def calculateSignificance():
    angles = [0,20,60,90]
    total_data1 = []
    total_data2 = []
    for angle in range(len(angles)):
        data1 = get_smoothed_emg_output(1, angles[angle], normalise=False)
        data2 = get_smoothed_emg_output(2, angles[angle], normalise=False)
        total_data1 = total_data1 + [data1]
        total_data2 = total_data2 + [data2]
    
    total_data1 = np.array(total_data1)
    total_data2 = np.array(total_data2)
    
    print("Position 1, significance between angles")
    avgs = []
    for muscle in range(5):
        a = []
        for angle in range(3):
            ds1 = np.mean(total_data1[angle,:,8000:12000,muscle], axis=1)
            ds2 = np.mean(total_data1[angle+1,:,8000:12000,muscle], axis=1)
            _,pt = stats.ttest_ind(ds1, ds2)
            a = a + [pt]
        avgs = avgs + [a]
    
    print(avgs)
    
    print("Position 2, significance between angles")
    
    avgs = []
    for muscle in range(5):
        a = []
        for angle in range(3):
            ds1 = np.mean(total_data2[angle,:,8000:12000,muscle], axis=1)
            ds2 = np.mean(total_data2[angle+1,:,8000:12000,muscle], axis=1)
            _,pt = stats.ttest_ind(ds1, ds2)
            a = a + [pt]
        avgs = avgs + [a]
    
    print(avgs)
    
    print("Significance between positions")
    
    avgs = []
    for muscle in range(5):
        a = []
        for angle in range(4):
            ds1 = np.mean(total_data1[angle,:,8000:12000,muscle], axis=1)
            ds2 = np.mean(total_data2[angle,:,8000:12000,muscle], axis=1)
            _,pt = stats.ttest_ind(ds1, ds2)
            a = a + [pt]
        avgs = avgs + [a]
    
    print(avgs)
    
    # baseline check for ST
    
    print("Position 1, significance between ST baslines")
    
    avgs = []
    for angle in range(3):
        ds1 = np.mean(np.concatenate([total_data1[angle,:,:4000,muscle],total_data1[angle,:,16000:,muscle]], axis=1), axis=1)
        ds2 = np.mean(np.concatenate([total_data1[angle+1,:,:4000,muscle],total_data1[angle+1,:,16000:,muscle]], axis=1), axis=1)
        _,pt = stats.ttest_ind(ds1, ds2)
        avgs = avgs + [pt]
    
    print(avgs)
        
        

# Plot the average per muscle of the smoothed and normalised EMG data
def plot_smoothed_emg_average_output(position, angle):
    data = get_smoothed_emg_output(position, angle, normalise=True)
    time_series_length = data.shape[1]
    num_coefficients = data.shape[2]
    rank = 2
    
    average_data = np.mean(data, axis=0)
    
    fig, axes = plt.subplots(1,num_coefficients,figsize=(8,8))
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.set_size_inches(18, 18)
    for i in range(num_coefficients):
        rects = axes[i].plot(average_data[:,i].tolist(), linewidth=3)
        axes[i].set_ylim([0,0.04])
    plt.show()
    
def plot_smoothed_emg_average_output_for_all_angles():
    fig, axes = plt.subplots(4,5,figsize=(8,8))
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.set_size_inches(12, 12)
    angles = [0,20,60,90]
    
    for angle in range(len(angles)):
        data = get_smoothed_emg_output(1, angles[angle], normalise=False)
        data2 = get_smoothed_emg_output(2, angles[angle], normalise=False)
        time_series_length = data.shape[1]
        num_coefficients = 5
        rank = 2
        
        average_data = np.mean(data, axis=0)
        std_data = np.std(data, axis=0)
        
        average_data2 = np.mean(data2, axis=0)
        std_data2 = np.std(data2, axis=0)
        for i in range(num_coefficients):
            axes[angle][i].fill_between(range(len(std_data[:,i].tolist())), average_data[:,i]-std_data[:,i], average_data[:,i]+std_data[:,i], color='#FF8888')
            axes[angle][i].fill_between(range(len(std_data2[:,i].tolist())), average_data2[:,i]-std_data2[:,i], average_data2[:,i]+std_data2[:,i], color='#8888FF')
            rects = axes[angle][i].plot(average_data[:,i].tolist(), linewidth=3, color='#FF0000')
            rects = axes[angle][i].plot(average_data2[:,i].tolist(), linewidth=3, color='#222288')
            axes[angle][i].set_ylim([0,0.07])
            axes[angle][i].tick_params(axis='both',which='both',left=False,bottom=False,labelbottom=False,labelleft=False,labelsize=12)
            axes[angle][i].spines["top"].set_visible(False)
            axes[angle][i].spines["right"].set_visible(False)
            axes[angle][i].spines["left"].set_visible(False)
    plt.show()
    
def plot_smoothed_emg_average_output_for_all_angles_normalised(position):
    fig, axes = plt.subplots(4,5,figsize=(8,8))
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.set_size_inches(12, 12)
    angles = [0,20,60,90]
    
    for angle in range(len(angles)):
        data = get_smoothed_emg_output(position, angles[angle], normalise=True)
        time_series_length = data.shape[1]
        num_coefficients = 5
        rank = 2
        
        average_data = np.mean(data, axis=0)
        for i in range(num_coefficients):
            #average_data[:,i] = amplitude_normalization(average_data[:,i])
            rects = axes[angle][i].plot(average_data[:,i].tolist(), linewidth=3)
            axes[angle][i].set_ylim([0,1.0])
    plt.show()

# Calculate the NMF of each EMG output for a given position and angle.
# iqr_limit: eliminate EMG trials where the inter-quartile range of 
#     synergy 1 vector values is greater than iqr_limit  
def get_NMF_output(position, angle, iqr_limit=None):
    
    data = get_smoothed_emg_output(position, angle, normalise=True)
    time_series_length = data.shape[1]
    num_coefficients = data.shape[2]
    rank = 2
    
    activation_patterns = np.empty([0,time_series_length,rank])
    contribution_vectors = np.empty([0,rank,num_coefficients])
    
    for i in range(data.shape[0]):
        a,b,c,rss=NMF_calculator(data[i,:time_series_length,:],rank=rank)
        
        iqr = stats.iqr(c[0])
        
        if not iqr_limit or iqr < iqr_limit:
            activation_patterns = np.append(activation_patterns, [a], axis=0)
            contribution_vectors = np.append(contribution_vectors, [c], axis=0)
        
    return activation_patterns, contribution_vectors
    
# Calculate the NMF of the average of EMG outputs for a given position and angle
def get_NMF_of_average_output(position, angle):
    data = get_smoothed_emg_output(position, angle, normalise=True)
    time_series_length = data.shape[1]
    num_coefficients = data.shape[2]
    rank = 2
    
    average_data = np.mean(data, axis=0)
    a,b,c,rss=NMF_calculator(average_data[:time_series_length],rank=rank)
    
    return a, c

# Average the EMG outputs then calculate 2 NMF synergies of the result and plot.
def plot_NMF_of_average_output(position, angle):
    activation_pattern, contribution_vector = get_NMF_of_average_output(position, angle)
    
    index = np.array(range(0,contribution_vector.shape[1])) - np.ones(contribution_vector.shape[1])*0.2 
    
    fig, axes = plt.subplots(2,2,figsize=(3,4))
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.set_size_inches(9, 6)
    
    rects = axes[0][0].plot(activation_pattern[:,0].tolist(), color='#222288', linewidth=3)
    rects = axes[0][1].bar(index,np.ravel(contribution_vector[0].tolist()), 0.4, color='#222288', error_kw=dict(lw=3, capsize=4, capthick=3), label='Position 1',capsize=4, alpha=1.0, edgecolor='#000000',linewidth=3)
    axes[0][1].set_ylim([0,6.0])
    
    rects = axes[1][0].plot(activation_pattern[:,1].tolist(), color='#222288', linewidth=3)
    rects = axes[1][1].bar(index,np.ravel(contribution_vector[1].tolist()), 0.4, color='#222288', error_kw=dict(lw=3, capsize=4, capthick=3), label='Position 1',capsize=4, alpha=1.0, edgecolor='#000000',linewidth=3)
    axes[0][1].set_ylim([0,6.0])
    
    plt.show()
    
def plot_NMF_of_average_output_all_angles(synergy, position):
    angles = [0,20,60,90]
    
    fig, axes = plt.subplots(2,4,figsize=(3,4))
    fig.subplots_adjust(top=0.95)
    fig.set_size_inches(12, 9)
    
    for angle in range(len(angles)):
        activation_pattern, contribution_vector = get_NMF_of_average_output(position, angles[angle])
        
        index = np.array(range(0,contribution_vector.shape[1])) - np.ones(contribution_vector.shape[1])*0.2 
        
        rects = axes[1][angle].plot(activation_pattern[:,synergy-1].tolist(), color='#222288', linewidth=3)
        rects = axes[0][angle].bar(index,np.ravel(contribution_vector[synergy-1].tolist()), 0.4, color='#222288', alpha=1.0, edgecolor='#000000',linewidth=3)
        #axes[0][angle].set_ylim([0,5.0])
    
    plt.show()

# Calculate two synergies for each EMG output and then average the results and plot.
def plot_average_NMF_output(position, angle, iqr_limit=None):
    activation_patterns, contribution_vectors = get_NMF_output(position, angle, iqr_limit)
    
    avg_activation_pattern = np.mean(activation_patterns, axis=0)
    avg_contribution_vector = np.mean(contribution_vectors, axis=0)
    
    sd_activation_pattern = np.std(activation_patterns, axis=0)
    sd_contribution_vector = np.std(contribution_vectors, axis=0)
    
    index = np.array(range(0,avg_contribution_vector.shape[1])) - np.ones(avg_contribution_vector.shape[1])*0.2 
    
    fig, axes = plt.subplots(2,2,figsize=(3,4))
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.set_size_inches(9, 6)
    
    axes[0][0].fill_between(range(len(avg_activation_pattern[:,0])), avg_activation_pattern[:,0]-sd_activation_pattern[:,0], avg_activation_pattern[:,0]+sd_activation_pattern[:,0], color='#8888FF')
    rects = axes[0][0].plot(avg_activation_pattern[:,0].tolist(), color='#222288', linewidth=3)
    rects = axes[0][1].bar(index,np.ravel(avg_contribution_vector[0].tolist()), 0.4, color='#222288', error_kw=dict(lw=3, capsize=4, capthick=3), yerr=sd_contribution_vector[0], label='Position 1',capsize=4, alpha=1.0, edgecolor='#000000',linewidth=3)
    axes[0][1].set_ylim([0,7.0])
    
    axes[1][0].fill_between(range(len(avg_activation_pattern[:,1])), avg_activation_pattern[:,1]-sd_activation_pattern[:,0], avg_activation_pattern[:,1]+sd_activation_pattern[:,1], color='#8888FF')
    rects = axes[1][0].plot(avg_activation_pattern[:,1].tolist(), color='#222288', linewidth=3)
    rects = axes[1][1].bar(index,np.ravel(avg_contribution_vector[1].tolist()), 0.4, color='#222288', error_kw=dict(lw=3, capsize=4, capthick=3), yerr=sd_contribution_vector[1], label='Position 1',capsize=4, alpha=1.0, edgecolor='#000000',linewidth=3)
    axes[1][1].set_ylim([0,7.0])
    
    plt.show()

def calculateNmfSignificance():

    positions = [1,2]
    angles = [0,20,60,90]
    
    for p in positions:
        all_contribution_vectors = []
        all_activation_patterns = []
        totals = []
        for a in range(len(angles)):
            
            activation_patterns, contribution_vectors = get_NMF_output(p, angles[a], 0.0)
            
            all_activation_patterns = all_activation_patterns + [np.array(activation_patterns)]
            all_contribution_vectors = all_contribution_vectors + [np.array(contribution_vectors)]
            
            tots = []
            for i in range(activation_patterns.shape[0]):
                tots = tots + [np.dot(activation_patterns[i,:,0:1] , contribution_vectors[i,0:1,:])]
            totals = totals + [tots]
        
        for m in range(5):
            ac = []    
            for a in range(len(angles)-1):
                ds1 = np.mean(np.array(totals)[a,:,8000:12000,m], axis=1)
                ds2 = np.mean(np.array(totals)[a+1,:,8000:12000,m], axis=1)
                
                _,pt = stats.ttest_ind(ds1, ds2)
                
                ac = ac + [pt]
            print(ac)
        
        for m in range(5):
            ps = []
            for a in range(len(angles)-1):
                _,pt = stats.ttest_ind(all_contribution_vectors[a][:,0,m], all_contribution_vectors[a+1][:,0,m])
                ps = ps + [pt] 
            
            print(ps)
            
        ac = []    
        for a in range(len(angles)-1):
            ds1 = np.mean(all_activation_patterns[a][:,8000:12000,0], axis=1)
            ds2 = np.mean(all_activation_patterns[a+1][:,8000:12000,0], axis=1)
            _,pt = stats.ttest_ind(ds1, ds2)
            
            ac = ac + [pt]
        print(ac)
        
def get_NMF_rank1_output(position, angle, iqr_limit=None):
    data = get_smoothed_emg_output(position, angle, normalise=False)
    time_series_length = data.shape[1]
    num_coefficients = data.shape[2]
    rank = 1
    
    activation_patterns = np.empty([0,time_series_length,rank])
    contribution_vectors = np.empty([0,rank,num_coefficients])
    
    rsss = []
    
    for i in range(data.shape[0]):
        a,b,c,rss=NMF_calculator_with_VAF_check(data[i,:time_series_length,:],rank=rank)
        
        rsss = rsss + [rss]
        
        iqr = stats.iqr(c[0])
        
        if not iqr_limit or iqr < iqr_limit:
            activation_patterns = np.append(activation_patterns, [a], axis=0)
            contribution_vectors = np.append(contribution_vectors, [c], axis=0)
    
    print("Avg RSS")
    print(np.mean(np.array(rsss), axis=0))
    return activation_patterns, contribution_vectors
    
def get_NMF_rank_output(rank, position, angle, iqr_limit=None, do_rss=False, _normalise=False):
    data = get_smoothed_emg_output(position, angle, normalise=_normalise)
    time_series_length = data.shape[1]
    num_coefficients = data.shape[2]
    
    activation_patterns = np.empty([0,time_series_length,rank])
    contribution_vectors = np.empty([0,rank,num_coefficients])
    
    rsss = np.empty([5,0])
    
    for i in range(data.shape[0]):
        if do_rss:
            a,b,c,rss=NMF_calculator_with_VAF_check(data[i,:time_series_length,:],rank=rank)
            rsss = np.append(rsss,np.reshape(rss,[5,1]), axis=1)
        else:
            a,b,c,rss=NMF_calculator(data[i,:time_series_length,:],rank=rank)
        
        iqr = stats.iqr(c[0])
        
        if not iqr_limit or iqr < iqr_limit:
            activation_patterns = np.append(activation_patterns, [a], axis=0)
            contribution_vectors = np.append(contribution_vectors, [c], axis=0)
    
    return activation_patterns, contribution_vectors, rsss

def plot_all_positions_angles_NMF_output(rank, synergy, iqr_limit=None, do_rss=False, normalise=False):
    positions = [2,1]
    angles = [0,20,60,90]
    
    fig, axes = plt.subplots(2,4)
    #fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.set_size_inches(12, 8)
    
    rsss = np.empty([5,0])
    
    if normalise:
        vector_y_lim = 7.3#1.0#
        vector_y_tick = 1.0#0.2#
        pattern_y_lim = 0.18#0.014#
    else:
        vector_y_lim = 1.0#6.0#
        vector_y_tick = 0.2#1.0#
        pattern_y_lim = 0.014#0.11#
    
    for p in positions:
        for a in range(len(angles)):
            
            activation_patterns, contribution_vectors, rss = get_NMF_rank_output(rank, p, angles[a], iqr_limit, do_rss,normalise)
            
            if do_rss:
                rsss = np.append(rsss, rss, axis=1)
                print(rsss.shape)
            
            avg_activation_pattern = np.mean(activation_patterns, axis=0)
            avg_contribution_vector = np.mean(contribution_vectors, axis=0)
            
            sd_activation_pattern = np.std(activation_patterns, axis=0)
            sd_contribution_vector = np.std(contribution_vectors, axis=0)
            
            index = np.array(range(0,avg_contribution_vector.shape[1])) - np.ones(avg_contribution_vector.shape[1])*0.2 
    
            axes[0][a].set_ylim([0,vector_y_lim])
            index_only = np.array(range(0,5))
            index = np.array(range(0,5)) - np.ones(5)*0.2 
            if not p == 1:
                index = np.array(range(0,5)) + np.ones(5)*0.2 
            axes[0][a].set_xticks(index_only)
            axes[0][a].set_xticklabels(['RF','VL','VM','ST','BF'])
            color='#FF0000'
            if p == 1:
                rects = axes[0][a].bar(index,np.ravel(avg_contribution_vector[synergy].tolist()), 0.4, color='#FF0000', error_kw=dict(lw=2, capsize=3, capthick=2), yerr=sd_contribution_vector[synergy], label='Position 1',capsize=3, alpha=1.0, edgecolor='#000000',linewidth=2, hatch="//")
            else:
                color = '#222288'
                rects = axes[0][a].bar(index,np.ravel(avg_contribution_vector[synergy].tolist()), 0.4, color='#222288', error_kw=dict(lw=2, capsize=3, capthick=2), yerr=sd_contribution_vector[synergy], label='Position 2',capsize=3, alpha=1.0, edgecolor='#000000',linewidth=2)
            rects[0].set_color(color)
            rects[1].set_color(color)
            rects[2].set_color(color)
            rects[3].set_color(color)
            rects[4].set_color(color)
            rects[0].set_edgecolor('#000000')
            rects[1].set_edgecolor('#000000')
            rects[2].set_edgecolor('#000000')
            rects[3].set_edgecolor('#000000')
            rects[4].set_edgecolor('#000000')
            axes[0][a].tick_params(axis='both',which='both',left=False,bottom=False,labelbottom=True,labelleft=False,labelsize=12)
            axes[0][a].yaxis.set_ticks(np.arange(0.0, vector_y_lim + 0.01, vector_y_tick))
            axes[0][a].spines["top"].set_visible(False)
            axes[0][a].spines["left"].set_visible(False)
            axes[0][a].spines["right"].set_visible(False)
            axes[0][a].plot()

            axes[1][a].set_ylim([0,pattern_y_lim])
            if p == 1:
                axes[1][a].fill_between(range(len(avg_activation_pattern[:,synergy])), avg_activation_pattern[:,synergy]-sd_activation_pattern[:,synergy], avg_activation_pattern[:,synergy]+sd_activation_pattern[:,synergy], color='#FF8888')
                rects = axes[1][a].plot(avg_activation_pattern[:,synergy].tolist(), color='#FF0000', linewidth=2)
            else:
                axes[1][a].fill_between(range(len(avg_activation_pattern[:,synergy])), avg_activation_pattern[:,synergy]-sd_activation_pattern[:,synergy], avg_activation_pattern[:,synergy]+sd_activation_pattern[:,synergy], color='#8888FF')
                rects = axes[1][a].plot(avg_activation_pattern[:,synergy].tolist(), color='#222288', linewidth=2)
            axes[1][a].tick_params(axis='both',which='both',left=False,bottom=False,labelbottom=False,labelleft=False,labelsize=12)
            axes[1][a].spines["top"].set_visible(False)
            axes[1][a].spines["right"].set_visible(False)
            axes[1][a].spines["left"].set_visible(False)
            axes[1][a].plot()
            
    plt.show()
    
    if do_rss:
        fig = plt.figure()
        print(np.mean(rsss, axis=1))
        plt.plot([1,2,3,4,5], np.mean(rsss, axis=1), color='#FF0000', linewidth=2)
        plt.fill_between([1,2,3,4,5], np.mean(rsss, axis=1)-np.std(rsss, axis=1), np.mean(rsss, axis=1)+np.std(rsss, axis=1), color='#FF8888')
        plt.show()

# Calculate synergy 1 for all angles and positions for each EMG output and plot the average.  
def plot_all_positions_angles_NMF_output_s1(iqr_limit=None):
    positions = [2,1]
    angles = [0,20,60,90]
    
    fig, axes = plt.subplots(2,4)
    #fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.set_size_inches(12, 8)
    
    for p in positions:
        for a in range(len(angles)):
            
            activation_patterns, contribution_vectors = get_NMF_rank1_output(p, angles[a], iqr_limit)
             
            avg_activation_pattern = np.mean(activation_patterns, axis=0)
            avg_contribution_vector = np.mean(contribution_vectors, axis=0)
            
            sd_activation_pattern = np.std(activation_patterns, axis=0)
            sd_contribution_vector = np.std(contribution_vectors, axis=0)
            
            index = np.array(range(0,avg_contribution_vector.shape[1])) - np.ones(avg_contribution_vector.shape[1])*0.2 
    
            axes[0][a].set_ylim([0,2.2])
            index_only = np.array(range(0,5))
            index = np.array(range(0,5)) - np.ones(5)*0.2 
            if not p == 1:
                index = np.array(range(0,5)) + np.ones(5)*0.2 
            axes[0][a].set_xticks(index_only)
            axes[0][a].set_xticklabels(['RF','VL','VM','ST','BF'])
            color='#FF0000'
            if p == 1:
                rects = axes[0][a].bar(index,np.ravel(avg_contribution_vector[0].tolist()), 0.4, color='#FF0000', error_kw=dict(lw=2, capsize=3, capthick=2), yerr=sd_contribution_vector[0], label='Position 1',capsize=3, alpha=1.0, edgecolor='#000000',linewidth=2, hatch="//")
            else:
                color = '#222288'
                rects = axes[0][a].bar(index,np.ravel(avg_contribution_vector[0].tolist()), 0.4, color='#222288', error_kw=dict(lw=2, capsize=3, capthick=2), yerr=sd_contribution_vector[0], label='Position 2',capsize=3, alpha=1.0, edgecolor='#000000',linewidth=2)
            rects[0].set_color(color)
            rects[1].set_color(color)
            rects[2].set_color(color)
            rects[3].set_color(color)
            rects[4].set_color(color)
            rects[0].set_edgecolor('#000000')
            rects[1].set_edgecolor('#000000')
            rects[2].set_edgecolor('#000000')
            rects[3].set_edgecolor('#000000')
            rects[4].set_edgecolor('#000000')
            axes[0][a].tick_params(axis='both',which='both',left=False,bottom=False,labelbottom=True,labelleft=False,labelsize=12)
            axes[0][a].yaxis.set_ticks(np.arange(0.0, 2.21, 0.5))
            axes[0][a].spines["top"].set_visible(False)
            axes[0][a].spines["left"].set_visible(False)
            axes[0][a].spines["right"].set_visible(False)
            axes[0][a].plot()

            axes[1][a].set_ylim([0,0.035])
            if p == 1:
                axes[1][a].fill_between(range(len(avg_activation_pattern[:,0])), avg_activation_pattern[:,0]-sd_activation_pattern[:,0], avg_activation_pattern[:,0]+sd_activation_pattern[:,0], color='#FF8888')
                rects = axes[1][a].plot(avg_activation_pattern[:,0].tolist(), color='#FF0000', linewidth=2)
            else:
                axes[1][a].fill_between(range(len(avg_activation_pattern[:,0])), avg_activation_pattern[:,0]-sd_activation_pattern[:,0], avg_activation_pattern[:,0]+sd_activation_pattern[:,0], color='#8888FF')
                rects = axes[1][a].plot(avg_activation_pattern[:,0].tolist(), color='#222288', linewidth=2)
            axes[1][a].tick_params(axis='both',which='both',left=False,bottom=False,labelbottom=False,labelleft=False,labelsize=12)
            axes[1][a].spines["top"].set_visible(False)
            axes[1][a].spines["right"].set_visible(False)
            axes[1][a].spines["left"].set_visible(False)
            axes[1][a].plot()
            
    plt.show()
    

# Calculate synergy 2 for all angles and positions for each EMG output and plot the average.   
def plot_all_positions_angles_NMF_output_s2(iqr_limit=None):
    positions = [2,1]
    angles = [0,20,60,90]
    
    fig, axes = plt.subplots(2,4)
    #fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.set_size_inches(12, 8)
    
    vector_y_lim = 6.0#1.0#
    vector_y_tick = 1.0#0.2#
    pattern_y_lim = 0.11#0.014#
    
    for p in positions:
        for a in range(len(angles)):
            activation_patterns, contribution_vectors = get_NMF_output(p, angles[a], iqr_limit)
            
            avg_activation_pattern = np.mean(activation_patterns, axis=0)
            avg_contribution_vector = np.mean(contribution_vectors, axis=0)
            
            sd_activation_pattern = np.std(activation_patterns, axis=0)
            sd_contribution_vector = np.std(contribution_vectors, axis=0)
            
            index = np.array(range(0,avg_contribution_vector.shape[1])) - np.ones(avg_contribution_vector.shape[1])*0.2 
    
            axes[0][a].set_ylim([0,vector_y_lim])
            index_only = np.array(range(0,5))
            index = np.array(range(0,5)) - np.ones(5)*0.2 
            if not p == 1:
                index = np.array(range(0,5)) + np.ones(5)*0.2 
            axes[0][a].set_xticks(index_only)
            axes[0][a].set_xticklabels(['RF','VL','VM','ST','BF'])
            color='#FF0000'
            if p == 1:
                rects = axes[0][a].bar(index,np.ravel(avg_contribution_vector[1].tolist()), 0.4, color='#FF0000', error_kw=dict(lw=2, capsize=3, capthick=2), yerr=sd_contribution_vector[1], label='Position 1',capsize=3, alpha=1.0, edgecolor='#000000',linewidth=2, hatch="//")
            else:
                color = '#222288'
                rects = axes[0][a].bar(index,np.ravel(avg_contribution_vector[1].tolist()), 0.4, color='#222288', error_kw=dict(lw=2, capsize=3, capthick=2), yerr=sd_contribution_vector[1], label='Position 2',capsize=3, alpha=1.0, edgecolor='#000000',linewidth=2)
            rects[0].set_color(color)
            rects[1].set_color(color)
            rects[2].set_color(color)
            rects[3].set_color(color)
            rects[4].set_color(color)
            rects[0].set_edgecolor('#000000')
            rects[1].set_edgecolor('#000000')
            rects[2].set_edgecolor('#000000')
            rects[3].set_edgecolor('#000000')
            rects[4].set_edgecolor('#000000')
            axes[0][a].tick_params(axis='both',which='both',left=False,bottom=False,labelbottom=True,labelleft=False,labelsize=12)
            axes[0][a].yaxis.set_ticks(np.arange(0.0, vector_y_lim + 0.01, vector_y_tick))
            axes[0][a].spines["top"].set_visible(False)
            axes[0][a].spines["left"].set_visible(False)
            axes[0][a].spines["right"].set_visible(False)
            axes[0][a].plot()

            axes[1][a].set_ylim([0,pattern_y_lim])
            if p == 1:
                axes[1][a].fill_between(range(len(avg_activation_pattern[:,1])), avg_activation_pattern[:,1]-sd_activation_pattern[:,1], avg_activation_pattern[:,1]+sd_activation_pattern[:,1], color='#FF8888')
                rects = axes[1][a].plot(avg_activation_pattern[:,1].tolist(), color='#FF0000', linewidth=2)
            else:
                axes[1][a].fill_between(range(len(avg_activation_pattern[:,1])), avg_activation_pattern[:,1]-sd_activation_pattern[:,1], avg_activation_pattern[:,1]+sd_activation_pattern[:,1], color='#8888FF')
                rects = axes[1][a].plot(avg_activation_pattern[:,1].tolist(), color='#222288', linewidth=2)
            axes[1][a].tick_params(axis='both',which='both',left=False,bottom=False,labelbottom=False,labelleft=False,labelsize=12)
            axes[1][a].spines["top"].set_visible(False)
            axes[1][a].spines["right"].set_visible(False)
            axes[1][a].spines["left"].set_visible(False)
            axes[1][a].plot()
            
    plt.show()

def simulation_file_picker(directory):
    files=[]
    for file in sorted(os.listdir(directory)):
        if file.endswith(".csv"):
            files.append(file)
    wholefile=[]
    for file in files:        
        wholefile.append(np.genfromtxt(os.path.join(directory, file),delimiter=','))
    return wholefile
    
def get_simulation_data():
    simulation_raw_directory="raw"
    simulation_vector_directory="vector"
    simulation_coefficients_directory="coeffs"

    raw = np.asarray(simulation_file_picker(simulation_raw_directory))
    vector = np.asarray(simulation_file_picker(simulation_vector_directory))
    coefficents = np.asarray(simulation_file_picker(simulation_coefficients_directory))
    
    return raw, vector, coefficents

# Plot NMF synergy 1 of position 1 compared to the simulation output   
def plot_position_1_simulation_comparison_s1(iqr_limit=0.5):
    angles = [0,20,60,90]
    
    fig, axes = plt.subplots(2,4)
    #fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.set_size_inches(12, 8)
    
    sim_raw, sim_vec, sim_coeff = get_simulation_data()
    
    for a in range(len(angles)):
        
        activation_patterns, contribution_vectors = get_NMF_output(1, angles[a], iqr_limit)
        
        avg_activation_pattern = np.mean(activation_patterns, axis=0)
        avg_contribution_vector = np.mean(contribution_vectors, axis=0)
        
        sd_activation_pattern = np.std(activation_patterns, axis=0)
        sd_contribution_vector = np.std(contribution_vectors, axis=0)
        
        index = np.array(range(0,avg_contribution_vector.shape[1])) - np.ones(avg_contribution_vector.shape[1])*0.2 

        axes[0][a].set_ylim([0,6.9])
        index_only = np.array(range(0,5))
        index = np.array(range(0,5)) - np.ones(5)*0.2 
        index2 = np.array(range(0,5)) + np.ones(5)*0.2 
        axes[0][a].set_xticks(index_only)
        axes[0][a].set_xticklabels(['RF','VL','VM','ST','BF'])
        color='#FF0000'
        rects = axes[0][a].bar(index,np.ravel(avg_contribution_vector[0].tolist()), 0.4, color='#FF0000', error_kw=dict(lw=2, capsize=3, capthick=2), yerr=sd_contribution_vector[0], label='Position 1',capsize=3, alpha=1.0, edgecolor='#000000',linewidth=2, hatch="//")
        
        color = '#222288'
        rects = axes[0][a].bar(index2,np.ravel(sim_vec[a][:,0].tolist()), 0.4, color='#222288', label='Position 2',capsize=3, alpha=1.0, edgecolor='#000000',linewidth=2)
        
        rects[0].set_color(color)
        rects[1].set_color(color)
        rects[2].set_color(color)
        rects[3].set_color(color)
        rects[4].set_color(color)
        rects[0].set_edgecolor('#000000')
        rects[1].set_edgecolor('#000000')
        rects[2].set_edgecolor('#000000')
        rects[3].set_edgecolor('#000000')
        rects[4].set_edgecolor('#000000')
        axes[0][a].tick_params(axis='both',which='both',left=False,bottom=False,labelbottom=True,labelleft=False,labelsize=12)
        axes[0][a].yaxis.set_ticks(np.arange(0.0, 6.91, 1.0))
        axes[0][a].spines["top"].set_visible(False)
        axes[0][a].spines["right"].set_visible(False)
        axes[0][a].spines["left"].set_visible(False)
        axes[0][a].plot()

        axes[1][a].set_ylim([0,0.2])
        step = 8.0 / len(avg_activation_pattern[:,0].tolist())
        data_times = [a*step for a in range(len(avg_activation_pattern[:,0].tolist()))]
        axes[1][a].fill_between(data_times, avg_activation_pattern[:,0]-sd_activation_pattern[:,0], avg_activation_pattern[:,0]+sd_activation_pattern[:,0], color='#FF8888')
        rects = axes[1][a].plot(data_times, avg_activation_pattern[:,0].tolist(), color='#FF0000', linewidth=2)

        step = 8.0 / len(sim_coeff[a][:,0].tolist())
        data_times = [a*step for a in range(len(sim_coeff[a][:,0].tolist()))]
        dat = [a for a in (sim_coeff[a][:,0].tolist())]
        rects = axes[1][a].plot(data_times, dat, color='#222288', linewidth=2)
        
        axes[1][a].tick_params(axis='both',which='both',left=False,bottom=False,labelbottom=False,labelleft=False,labelsize=12)
        axes[1][a].spines["top"].set_visible(False)
        axes[1][a].spines["right"].set_visible(False)
        axes[1][a].spines["left"].set_visible(False)
        axes[1][a].plot()
            
    plt.show()
    
# Plot the defined position/angle time series for each muscle per participant (only certain participants chosen)
# check_smoothed_raw(position, angle, normalise=True)
#check_smoothed_raw(2, 0)

# Plot the average time series across all participants for each angle and position
#plot_smoothed_emg_average_output_for_all_angles()

# Calculate the significance between each time series plotted above
#calculateSignificance()

# Calculate the NMF for each time series/action and average for 
plot_NMF_of_average_output_all_angles(1,1)
#plot_NMF_of_average_output_all_angles(2,1)


#plot_all_positions_angles_NMF_output_s2(iqr_limit=0.0)
#plot_all_positions_angles_NMF_output_s1(iqr_limit=None)
#plot_all_positions_angles_NMF_output(2,0,iqr_limit=None, do_rss=False, normalise=True)
#calculateNmfSignificance()
#plot_position_1_simulation_comparison_s1(iqr_limit=0.5)

#plot_smoothed_emg_average_output_for_all_angles(1)