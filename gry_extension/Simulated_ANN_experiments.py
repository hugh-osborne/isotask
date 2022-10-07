#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:09:35 2019

@author: gareth
"""
from __future__ import division
import numpy as np
import tkinter as tk
import os
import matplotlib.pyplot as plt 
import keras
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Activation 
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

(trainX, testX, trainY, testY) = train_test_split(labels,stimulation_combinations_cortical_combined, test_size=0.25)

model = Sequential()
model.add(Dense(1000,input_shape=(15,), activation="tanh"))
model.add(LeakyReLU(alpha=0.1))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(2))
model.add(Activation((lambda x: relu(x, max_value=1))))            

# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.001
EPOCHS = 1000

opt = optimizers.SGD(lr=INIT_LR)
model.compile(loss="mean_absolute_error", optimizer=opt,)

H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=10)
score = model.evaluate(testX, testY, verbose=0)

plt.plot(H.history['loss'], label='train')  
plt.plot(H.history['val_loss'], label='test')
plt.legend()
plt.show()

answer = []
example = 0
for example in labels:  
    answer.append(model.predict(np.reshape(example,([1,15]))))
answer = np.round(np.reshape((answer),[len(files)*inflator,2]),2)
answer = answer*stim_com_max
stimulation_combinations_cortical_combined = np.transpose(np.asarray([(0,50,100,150,200,250,300,350,400,400,400,400,400,400,400,400,400,400,400,450,500,550)*inflator,(400,400,400,400,400,400,400,400,0,50,100,150,200,250,300,350,450,500,550,400,400,400)*inflator]))
plt.plot(answer)
plt.show()
plt.plot(stimulation_combinations_cortical_combined)

#Peripheral
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

stimulation_combinations_peripheral = []
for file in files:            
    stimulation_combinations_peripheral.append(list(map(float,file[7:-4].split(sep="_"))))
stimulation_combinations_peripheral = np.asarray(stimulation_combinations_peripheral*inflator)

stim_com_max=np.amax(stimulation_combinations_peripheral)
stimulation_combinations_peripheral = stimulation_combinations_peripheral/stim_com_max
stimulation_combinations_peripheral = np.round(np.float32(stimulation_combinations_peripheral),2)

(trainX, testX, trainY, testY) = train_test_split(labels,stimulation_combinations_peripheral, test_size=0.25)
model = Sequential()
model.add(Dense(1000,input_shape=(15,), activation="tanh"))
model.add(LeakyReLU(alpha=0.1))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(5))
model.add(Activation((lambda x: relu(x, max_value=1))))            

# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.001
EPOCHS = 100

opt = optimizers.SGD(lr=INIT_LR)
model.compile(loss="mean_absolute_error", optimizer=opt,)

H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=10)
score = model.evaluate(testX, testY, verbose=0)

plt.plot(H.history['loss'], label='train')  
plt.plot(H.history['val_loss'], label='test')
plt.legend()
plt.show()

answer = []
example = 0
for example in labels:  
    answer.append(model.predict(np.reshape(example,([1,15]))))
answer = np.round(np.reshape((answer),[len(files)*inflator,5]),5)
answer = answer*stim_com_max

stimulation_combinations_peripheral=[]
for file in files:            
    stimulation_combinations_peripheral.append(list(map(float,file[7:-4].split(sep="_"))))
stimulation_combinations_peripheral = np.asarray(stimulation_combinations_peripheral*inflator)

plt.plot(answer)
plt.show()
plt.plot(stimulation_combinations_peripheral)

