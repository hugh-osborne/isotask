# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:57:29 2019

@author: bsgjry
"""
import numpy as np
import matplotlib.pyplot as plt
import nimfa
import tkinter as tk
import os
from tkinter import filedialog
from keras.models import Sequential
from keras.layers import Dense,LeakyReLU,Activation 
from keras.activations import relu
from keras import optimizers
from sklearn.model_selection import train_test_split

def open_file():
    root = tk.Tk()
    root.withdraw()
    file_path = tk.filedialog.askopenfilename()
    wholefile=np.genfromtxt(file_path,delimiter=',')
    return wholefile

def select_directory():
    root = tk.Tk()
    root.withdraw()
    directory = tk.filedialog.askdirectory()
    return directory

def file_generator():
    directory=select_directory()
    files=[]
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            files.append(file)    
    wholefile=[]
    inc=0
    for file in files:
        inc+=1
        print(inc,"/",len(files))
        wholefile.append(np.genfromtxt(os.path.join(directory, file),delimiter=','))    
    wholefile=np.asarray(wholefile)
    return wholefile

def amplitude_normalization(bwah):
    for i in range(5):
        bwah[i] += np.random.normal(0,1,bwah[i].shape)*0
        xmax, xmin = bwah[i].max(), bwah[i].min()
        bwah[i] = bwah[i]/xmax
    bwah = np.absolute(bwah)
    return bwah

def rms_calc(data):
    data = np.asarray(data)
    squared_sum = np.sum(data*data)
    rms = np.sqrt(squared_sum/len(data))
    return rms

#Waveform length
def waveform_length_calculator(data):
    waveform_length=[]
    for index in range(1,len(data)-1):
        waveform_length.append((data[index]-data[index-1]))
    waveform_length=np.sum(np.absolute(waveform_length))
    return waveform_length

def NMF_calculator(x,rank):
    nmf = nimfa.Nmf(abs(x), seed="nndsvd",  rank=rank, max_iter=500)    
    nmf_fit = nmf()
    print('Evar: %5.4f' % nmf_fit.fit.evar())
    vector=np.asarray(nmf_fit.basis())
    synergy_coeff=nmf_fit.fit.coef()
    return vector,synergy_coeff 

#Individual NMF runs and reset the H or W weights after each run
def NMF_reconstruction(x,rank,w,h):
    nmf = nimfa.Nmf(abs(x), W=w,H=h, rank=rank, max_iter=1)    
    nmf_fit = nmf()
    print('Evar: %5.4f' % nmf_fit.fit.evar())
    vector=np.asarray(nmf_fit.basis())
    synergy_coeff=nmf_fit.fit.coef()
    return vector,synergy_coeff 

def emg_parameters(h,w):
    v = np.matmul(w,h)
    label=[]
    maxamp = []
    rms = []
    waveformlen = []
    for muscle in range(5):
        maxamp.append(np.amax(v[:,muscle]))
        waveformlen.append(waveform_length_calculator(v[:,muscle]))
        rms.append(rms_calc(v[:,muscle]))
    label = [maxamp,rms,waveformlen]
    label = np.reshape(np.asarray(label).transpose(),(int(len(label)*len(label[0])/15),15))
    for row in range(label.shape[0]):
        label[row,:] = label[row,:]/np.max(label[row,:])
    return v,label

healthy_0_coeff=open_file()
healthy_90_coeff=open_file()
healthy_0_vector=open_file()
healthy_90_vector=open_file()

coefficient_cortical_dir=select_directory()
coefficient_peripheral_dir=select_directory()
vector_cortical_dir=select_directory()
vector_peripheral_dir=select_directory()

#0:3 extensor 4:7 flexor
#0:2 partial 2:4 total
#0=0,1=90

coefficient_cortical = []
coefficient_peripheral = []
vector_cortical = []
vector_peripheral = []

files1=[]
for files in os.listdir(coefficient_cortical_dir):
    if files.endswith(".csv"):
        files1.append(files)
for files in files1:        
    coefficient_cortical.append(np.genfromtxt(os.path.join(coefficient_cortical_dir, files),delimiter=','))

files2=[]
for file in os.listdir(coefficient_peripheral_dir):
    if file.endswith(".csv"):
        files2.append(file)
for file in files2:        
    coefficient_peripheral.append(np.genfromtxt(os.path.join(coefficient_peripheral_dir, file),delimiter=','))

files3=[]
for file in os.listdir(vector_cortical_dir):
    if file.endswith(".csv"):
        files3.append(file)
for file in files3:        
    vector_cortical.append(np.genfromtxt(os.path.join(vector_cortical_dir, file),delimiter=','))

files4=[]
for file in os.listdir(vector_peripheral_dir):
    if file.endswith(".csv"):
        files4.append(file)
for file in files4:        
    vector_peripheral.append(np.genfromtxt(os.path.join(vector_peripheral_dir, file),delimiter=','))


filename = 'Healthy_vs_injury_total_extensor'
color1='black'
color2='red'
names=["Healthy","Cortical Injury","Peripheral Injury"]
nrow = 5
ncol = 4
fig, axs = plt.subplots(nrow, ncol,figsize=(19,15))
i=0
color2 = 'red'
for i, ax in enumerate(fig.axes):
    if i == 0:
        ax.bar((1,2,3,4,5),(healthy_0_vector[:,0]),color=color1,edgecolor=color1)
        ax.bar((9,10,11,12,13),(healthy_0_vector[:,1]),color=color2,edgecolor=color1)      
        ax.set_title("0$^\circ$",weight="bold",fontsize=12.5)
        ax.set_ylabel('Vector\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
        ax.set_xticks((1,2,3,4,5,9,10,11,12,13))
        ax.set_xticklabels((1,2,3,4,5,1,2,3,4,5))
        ax.text(-6.5,2.5,"Healthy",fontsize=14,clip_on=False,weight="bold")
    if i ==1:           
        ax.bar((1,2,3,4,5),(healthy_90_vector[:,0]),color=color1,edgecolor=color1)
        ax.bar((9,10,11,12,13),(healthy_90_vector[:,1]),color=color2,edgecolor=color1)
        ax.set_xticks((1,2,3,4,5,9,10,11,12,13))
        ax.set_xticklabels((1,2,3,4,5,1,2,3,4,5))
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
        ax.legend(("Synergy 1","Synergy 2"),loc=(0.75,0.8))
        ax.set_title("90$^\circ$",weight="bold",fontsize=12.5)
    if i ==2:
        ax.plot(healthy_0_coeff[:,0],color=color1)
        ax.plot(healthy_0_coeff[:,1],color=color2)
        ax.set_title("0$^\circ$",weight="bold",fontsize=12.5)
        ax.set_ylabel('Coefficient\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))
    if i ==3:    
        ax.plot(healthy_90_coeff[:,0],color=color1)
        ax.plot(healthy_90_coeff[:,1],color=color2)
        ax.legend(("Synergy 1","Synergy 2"),loc=(0.85,0.8))
        ax.set_title("90$^\circ$",weight="bold",fontsize=12.5)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))
    if i ==4:
        ax.bar((1,2,3,4,5),(vector_cortical[0]),color=color1,edgecolor=color1)
        ax.set_xticks((1,2,3,4,5))
        ax.set_xticklabels((1,2,3,4,5))
        ax.text(-2.5,2.5,"Cortical",fontsize=14,clip_on=False,weight="bold")
        ax.text(-2,2,"Partial",fontsize=14,clip_on=False,weight="bold")
        ax.set_ylabel('Vector\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
    if i ==5:
        ax.bar((1,2,3,4,5),(vector_cortical[1]),color=color1,edgecolor=color1)
        ax.set_xticks((1,2,3,4,5))
        ax.set_xticklabels((1,2,3,4,5))
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
    if i ==6:
        ax.plot((coefficient_cortical[0]),color=color1)
        ax.set_ylabel('Coefficient\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))
    if i ==7:
        ax.plot((coefficient_cortical[1]),color=color1)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))
    if i ==8:
        ax.bar((1,2,3,4,5),(vector_cortical[2][:,0]),color=color1,edgecolor=color1)
        ax.bar((9,10,11,12,13),(vector_cortical[2][:,1]),color=color2,edgecolor=color1)
        ax.set_xticks((1,2,3,4,5,9,10,11,12,13))
        ax.set_xticklabels((1,2,3,4,5,1,2,3,4,5))
        ax.text(-5.5,2,"Total",fontsize=14,clip_on=False,weight="bold")
        ax.set_ylabel('Vector\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
    if i ==9:
        ax.bar((1,2,3,4,5),(vector_cortical[3][:,0]),color=color1,edgecolor=color1)
        ax.bar((9,10,11,12,13),(vector_cortical[3][:,1]),color=color2,edgecolor=color1)
        ax.set_xticks((1,2,3,4,5,9,10,11,12,13))
        ax.set_xticklabels((1,2,3,4,5,1,2,3,4,5))
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
    if i ==10:
        ax.plot((coefficient_cortical[2][:,0]),color=color1)
        ax.plot((coefficient_cortical[2][:,1]),color=color2)
        ax.set_ylabel('Coefficient\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))
    if i ==11:
        ax.plot((coefficient_cortical[3][:,0]),color=color1)        
        ax.plot((coefficient_cortical[3][:,1]),color=color2)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))        
    if i ==12:
        ax.bar((1,2,3,4,5),(vector_peripheral[0]),color=color1,edgecolor=color1)
        ax.set_xticks((1,2,3,4,5))
        ax.set_xticklabels((1,2,3,4,5))
        ax.text(-2.5,2.5,"Peripheral",fontsize=14,clip_on=False,weight="bold")
        ax.text(-2,2,"Partial",fontsize=14,clip_on=False,weight="bold")
        ax.set_ylabel('Vector\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
    if i ==13:
        ax.bar((1,2,3,4,5),(vector_peripheral[1]),color=color1,edgecolor=color1)
        ax.set_xticks((1,2,3,4,5))
        ax.set_xticklabels((1,2,3,4,5))
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
    if i ==14:
        ax.plot((coefficient_peripheral[0]),color=color1)
        ax.set_ylabel('Coefficient\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))
    if i ==15:
        ax.plot((coefficient_peripheral[1]),color=color1)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))
    if i ==16:
        ax.bar((1,2,3,4,5),(vector_peripheral[2][:,0]),color=color1,edgecolor=color1)
        ax.bar((9,10,11,12,13),(vector_peripheral[2][:,1]),color=color2,edgecolor=color1)
        ax.set_xticks((1,2,3,4,5,9,10,11,12,13))
        ax.set_xticklabels((1,2,3,4,5,1,2,3,4,5))
        ax.text(-5.5,2,"Total",fontsize=14,clip_on=False,weight="bold")
        ax.set_ylabel('Vector\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)             
    if i ==17:
        ax.bar((1,2,3,4,5),(vector_peripheral[3][:,0]),color=color1,edgecolor=color1)
        ax.bar((9,10,11,12,13),(vector_peripheral[3][:,1]),color=color2,edgecolor=color1)
        ax.set_xticks((1,2,3,4,5,9,10,11,12,13))
        ax.set_xticklabels((1,2,3,4,5,1,2,3,4,5))
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
    if i ==18:
        ax.plot((coefficient_peripheral[2][:,0]),color=color1)
        ax.plot((coefficient_peripheral[2][:,1]),color=color2)
        ax.set_ylabel('Coefficient\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))
    if i ==19:
        ax.plot((coefficient_peripheral[3][:,0]),color=color1)        
        ax.plot((coefficient_peripheral[3][:,1]),color=color2)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False) 
    ax.tick_params(axis='both',which='both',bottom=False,left=False,labelleft=False,labelsize=12.5)  
plt.tight_layout()    
fig.savefig((filename+".tiff"))
fig.savefig((filename+".svg")) 
    
filename = 'Healthy_vs_injury_total_flexor'
color1='black'
color2='red'
names=["Healthy","Cortical Injury","Peripheral Injury"]
nrow = 5
ncol = 4
fig, axs = plt.subplots(nrow, ncol,figsize=(19,15))
i=0
color2 = 'red'
for i, ax in enumerate(fig.axes):
    if i == 0:
        ax.bar((1,2,3,4,5),(healthy_0_vector[:,0]),color=color1,edgecolor=color1)
        ax.bar((9,10,11,12,13),(healthy_0_vector[:,1]),color=color2,edgecolor=color1)      
        ax.set_title("0$^\circ$",weight="bold",fontsize=12.5)
        ax.set_ylabel('Vector\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
        ax.set_xticks((1,2,3,4,5,9,10,11,12,13))
        ax.set_xticklabels((1,2,3,4,5,1,2,3,4,5))
        ax.text(-6.5,2.5,"Healthy",fontsize=14,clip_on=False,weight="bold")
    if i ==1:           
        ax.bar((1,2,3,4,5),(healthy_90_vector[:,0]),color=color1,edgecolor=color1)
        ax.bar((9,10,11,12,13),(healthy_90_vector[:,1]),color=color2,edgecolor=color1)
        ax.set_xticks((1,2,3,4,5,9,10,11,12,13))
        ax.set_xticklabels((1,2,3,4,5,1,2,3,4,5))
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
        ax.legend(("Synergy 1","Synergy 2"),loc=(0.75,0.8))
        ax.set_title("90$^\circ$",weight="bold",fontsize=12.5)
    if i ==2:
        ax.plot(healthy_0_coeff[:,0],color=color1)
        ax.plot(healthy_0_coeff[:,1],color=color2)
        ax.set_title("0$^\circ$",weight="bold",fontsize=12.5)
        ax.set_ylabel('Coefficient\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))
    if i ==3:    
        ax.plot(healthy_90_coeff[:,0],color=color1)
        ax.plot(healthy_90_coeff[:,1],color=color2)
        ax.legend(("Synergy 1","Synergy 2"),loc=(0.85,0.8))
        ax.set_title("90$^\circ$",weight="bold",fontsize=12.5)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))
    if i ==4:
        ax.bar((1,2,3,4,5),(vector_cortical[4]),color=color1,edgecolor=color1)
        ax.set_xticks((1,2,3,4,5))
        ax.set_xticklabels((1,2,3,4,5))
        ax.text(-2.5,2.5,"Cortical",fontsize=14,clip_on=False,weight="bold")
        ax.text(-2,2,"Partial",fontsize=14,clip_on=False,weight="bold")
        ax.set_ylabel('Vector\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
    if i ==5:
        ax.bar((1,2,3,4,5),(vector_cortical[5]),color=color1,edgecolor=color1)
        ax.set_xticks((1,2,3,4,5))
        ax.set_xticklabels((1,2,3,4,5))
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
    if i ==6:
        ax.plot((coefficient_cortical[4]),color=color1)
        ax.set_ylabel('Coefficient\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))
    if i ==7:
        ax.plot((coefficient_cortical[5]),color=color1)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))
    if i ==8:
        ax.bar((1,2,3,4,5),(vector_cortical[6][:,0]),color=color1,edgecolor=color1)
        ax.bar((9,10,11,12,13),(vector_cortical[6][:,1]),color=color2,edgecolor=color1)
        ax.set_xticks((1,2,3,4,5,9,10,11,12,13))
        ax.set_xticklabels((1,2,3,4,5,1,2,3,4,5))
        ax.text(-5.5,2,"Total",fontsize=14,clip_on=False,weight="bold")
        ax.set_ylabel('Vector\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
    if i ==9:
        ax.bar((1,2,3,4,5),(vector_cortical[7][:,0]),color=color1,edgecolor=color1)
        ax.bar((9,10,11,12,13),(vector_cortical[7][:,1]),color=color2,edgecolor=color1)
        ax.set_xticks((1,2,3,4,5,9,10,11,12,13))
        ax.set_xticklabels((1,2,3,4,5,1,2,3,4,5))
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
    if i ==10:
        ax.plot((coefficient_cortical[6][:,0]),color=color1)
        ax.plot((coefficient_cortical[6][:,1]),color=color2)
        ax.set_ylabel('Coefficient\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))
    if i ==11:
        ax.plot((coefficient_cortical[7][:,0]),color=color1)        
        ax.plot((coefficient_cortical[7][:,1]),color=color2)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))        
    if i ==12:
        ax.bar((1,2,3,4,5),(vector_peripheral[4]),color=color1,edgecolor=color1)
        ax.set_xticks((1,2,3,4,5))
        ax.set_xticklabels((1,2,3,4,5))
        ax.text(-2.5,2.5,"Peripheral",fontsize=14,clip_on=False,weight="bold")
        ax.text(-2,2,"Partial",fontsize=14,clip_on=False,weight="bold")
        ax.set_ylabel('Vector\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
    if i ==13:
        ax.bar((1,2,3,4,5),(vector_peripheral[5]),color=color1,edgecolor=color1)
        ax.set_xticks((1,2,3,4,5))
        ax.set_xticklabels((1,2,3,4,5))
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
    if i ==14:
        ax.plot((coefficient_peripheral[4]),color=color1)
        ax.set_ylabel('Coefficient\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))
    if i ==15:
        ax.plot((coefficient_peripheral[5]),color=color1)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))
    if i ==16:
        ax.bar((1,2,3,4,5),(vector_peripheral[6][:,0]),color=color1,edgecolor=color1)
        ax.bar((9,10,11,12,13),(vector_peripheral[6][:,1]),color=color2,edgecolor=color1)
        ax.set_xticks((1,2,3,4,5,9,10,11,12,13))
        ax.set_xticklabels((1,2,3,4,5,1,2,3,4,5))
        ax.text(-5.5,2,"Total",fontsize=14,clip_on=False,weight="bold")
        ax.set_ylabel('Vector\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)             
    if i ==17:
        ax.bar((1,2,3,4,5),(vector_peripheral[7][:,0]),color=color1,edgecolor=color1)
        ax.bar((9,10,11,12,13),(vector_peripheral[7][:,1]),color=color2,edgecolor=color1)
        ax.set_xticks((1,2,3,4,5,9,10,11,12,13))
        ax.set_xticklabels((1,2,3,4,5,1,2,3,4,5))
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
    if i ==18:
        ax.plot((coefficient_peripheral[6][:,0]),color=color1)
        ax.plot((coefficient_peripheral[6][:,1]),color=color2)
        ax.set_ylabel('Coefficient\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))
    if i ==19:
        ax.plot((coefficient_peripheral[7][:,0]),color=color1)        
        ax.plot((coefficient_peripheral[7][:,1]),color=color2)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))   
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False) 
    ax.tick_params(axis='both',which='both',bottom=False,left=False,labelleft=False,labelsize=12.5)  
plt.tight_layout()    
fig.savefig((filename+".tiff"))
fig.savefig((filename+".svg"))     
        
#0:3 extensor 4:7 flexor
#0:2 partial 2:4 total
#0=0,1=90

#Reconstruction
filename = 'Healthy_vs_injury_type'
color1='black'
color2='red'
names=["Healthy","Peripheral Injury","Cortical Injury"]
nrow = 2
ncol = 3
fig, axs = plt.subplots(nrow, ncol,figsize=(12,10))
i=0
color2 = 'red'
for i, ax in enumerate(fig.axes):
    if i ==0:
        ax.bar((1,2,3,4,5),(healthy_0_vector[:,0]),color=color1,edgecolor=color1)
        ax.bar((9,10,11,12,13),(healthy_0_vector[:,1]),color=color2,edgecolor=color1)  
        ax.set_title(names[0],weight="bold",fontsize=12.5)
        ax.set_ylabel('Vector Normalized \n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
        ax.set_xticks((1,2,3,4,5,9,10,11,12,13))
        ax.set_xticklabels((1,2,3,4,5,1,2,3,4,5))
        ax.legend(("Synergy 1","Synergy 2"),loc=(0.65,0.8))
    if i ==1:
        ax.bar((1,2,3,4,5),(vector_peripheral[2][:,0]),color=color1,edgecolor=color1)
        ax.bar((9,10,11,12,13),(vector_peripheral[2][:,1]),color=color2,edgecolor=color1)
        ax.set_title(names[1],weight="bold",fontsize=12.5)
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
        ax.set_xticks((1,2,3,4,5,9,10,11,12,13))
        ax.set_xticklabels((1,2,3,4,5,1,2,3,4,5))
    if i ==2:
        ax.bar((1,2,3,4,5),(vector_cortical[2][:,0]),color=color1,edgecolor=color1)
        ax.bar((9,10,11,12,13),(vector_cortical[2][:,1]),color=color2,edgecolor=color1)
        ax.set_title(names[2],weight="bold",fontsize=12.5)
        ax.set_xlabel('Channels',weight="bold",fontsize=12.5)
        ax.set_xticks((1,2,3,4,5,9,10,11,12,13))
        ax.set_xticklabels((1,2,3,4,5,1,2,3,4,5))
    if i ==3:
        ax.plot(healthy_0_coeff[:,0],color=color1)
        ax.plot(healthy_0_coeff[:,1],color=color2)
        ax.set_ylabel('Coefficient Normalized \n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))
    if i ==4:
        ax.plot(coefficient_peripheral[2][:,0],color=color1)
        ax.plot(coefficient_peripheral[2][:,1],color=color2)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))
    if i ==5:
        ax.plot(coefficient_cortical[2][:,0],color=color1)
        ax.plot(coefficient_cortical[2][:,1],color=color2)
        ax.set_xlabel('Time /s',weight="bold",fontsize=12.5)
        ax.set_xticks((0,4000))
        ax.set_xticklabels((0,8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False) 
    ax.tick_params(axis='both',which='both',bottom=False,left=False,labelleft=True,labelsize=12.5)  
plt.tight_layout()
fig.savefig((filename+".tiff"))
fig.savefig((filename+".svg"))

healthy_h = healthy_0_vector
healthy_w = healthy_0_coeff


#Fixed Vector
#vector = W, coeff = H  
reconstructed_synergy_coeff=healthy_h
for _ in range(0,500):
    vector,reconstructed_synergy_coeff = NMF_reconstruction(peripheral_0_raw,rank=2,w=healthy_w,h=reconstructed_synergy_coeff)

#Fixed Coeff
#vector = W, coeff = H  
reconstructed_vector=healthy_w
for _ in range(0,500):
    reconstructed_vector,synergy_coeff = NMF_reconstruction(peripheral_0_raw,rank=2,w=vector,h=healthy_h)

#Cortical
directory=select_directory()
files=[]
for file in os.listdir(directory):
    if file.endswith(".csv"):
        files.append(file)

wholefile=[]
inc=0
for file in files:
    inc+=1
    print(inc,"/",len(files))
    wholefile.append(np.genfromtxt(os.path.join(directory, file),delimiter=',',skip_header=1))    
wholefile=np.asarray(wholefile)

labels=[]
maxamp = []
rms = []
waveformlen = []
for sample in range(0,len(wholefile)):
    for muscle in range(5):
        maxamp.append(np.amax(wholefile[sample,:,muscle]))
        waveformlen.append(waveform_length_calculator(wholefile[sample,:,muscle]))
        rms.append(rms_calc(wholefile[sample,:,muscle]))

inflator = 1000
labels = [maxamp*inflator,rms*inflator,waveformlen*inflator]
labels = np.reshape(np.asarray(labels).transpose(),(int(len(labels)*len(labels[0])/15),15))
for row in range(labels.shape[0]):
    labels[row,:] = labels[row,:]/np.max(labels[row,:])
labels = np.round(np.float32(labels),2)


stimulation_combinations_cortical_combined = np.transpose(np.asarray([(0,50,100,150,200,250,300,350,400,400,400,400,400,400,400,400,400,400,400,400,450,500,550)*inflator,(400,400,400,400,400,400,400,400,0,50,100,150,200,250,300,350,400,450,500,550,400,400,400)*inflator]))
stim_com_max=np.amax(stimulation_combinations_cortical_combined)
stimulation_combinations_cortical_combined = stimulation_combinations_cortical_combined/stim_com_max
stimulation_combinations_cortical_combined = np.round(np.float32(stimulation_combinations_cortical_combined),2)

stimulation_combinations_cortical_combined_copy=np.copy(stimulation_combinations_cortical_combined)
label_copy = np.copy(labels)


(trainX, testX, trainY, testY) = train_test_split(labels,stimulation_combinations_cortical_combined, test_size=0.25)
model = Sequential()
model.add(Dense(1000,input_shape=(15,), activation="tanh"))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(2))
model.add(Activation((lambda x: relu(x, max_value=1))))            

# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.001
EPOCHS = 1000

opt = optimizers.SGD(lr=INIT_LR)
model.compile(loss="mean_absolute_error", optimizer=opt,)

H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=32)
score = model.evaluate(testX, testY, verbose=0)

filename='Network_training'
nrow = 1
ncol = 1
fig, axs = plt.subplots(nrow, ncol,sharex=('row'),figsize=(8,8))
color1 = 'black'
color2 = 'red'

for i, ax in enumerate(fig.axes):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel('Loss /arb. unti',weight="bold",fontsize=12.5)
    ax.set_xlabel('Training Epoch',weight="bold",fontsize=12.5)
    ax.tick_params(axis='both',which='both',labelsize=12.5)  
    ax.plot(H.history['loss'],label="Train",color = color1)
    ax.plot(H.history['val_loss'],label="Test",color = color2)
    ax.legend(fontsize=12.5)
fig.savefig((filename+".tiff"))
fig.savefig((filename+".svg"))

answer = []
example = 0
for example in range(0,len(label_copy)):
    _ = label_copy[example:example+1]
    answer.append(model.predict(_))
answer = np.reshape(np.asarray(answer),[404,2])


filename = 'Reconstructed_synergy'
color1='black'
color2='red'
names=["Healthy","Cortical Injury","Peripheral Injury"]
nrow = 1
ncol = 2
fig, axs = plt.subplots(nrow, ncol,figsize=(12,12))
i=0
color2 = 'red'
for i, ax in enumerate(fig.axes):
    if i == 0:
        ax.bar((1,2,3,4,5),(healthy_0_vector[:,0]),color=color1,edgecolor=color1)
        ax.bar((9,10,11,12,13),(healthy_0_vector[:,1]),color=color2,edgecolor=color1)      
        ax.set_ylabel('Vector\n Amplitude /arb. units',weight="bold",fontsize=12.5)
        ax.set_xticks((1,2,3,4,5,9,10,11,12,13))
        ax.set_xticklabels((1,2,3,4,5,1,2,3,4,5))
    if i ==1:
        ax.plot(healthy_0_coeff[:,0],color=color1)
        ax.plot(healthy_0_coeff[:,1],color=color2)
        ax.set_ylabel('Coefficient\n Amplitude /arb. units',weight="bold",fontsize=12.5)           
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='both',which='both',bottom=True,labelbottom=True,labelsize=12.5)  
fig.savefig((filename+".tiff"))
fig.savefig((filename+".svg"))


#Predictions
target = model.predict(healthy_label)*stim_com_max
current =  model.predict(reconstructed_label)*stim_com_max
differnce = target-current

#Feed difference into injured simulation 

