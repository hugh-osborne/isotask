# Once the Python Shared Library has been built in MIIND,
# copy this file to the results directory (where the .cpp and .so files were
# created).

import pylab
import numpy
import matplotlib.pyplot as plt
import imp
import datetime
from operator import add
from random import randint
import libisopy as miind
import nimfa

# Comment out MPI, comm and rank lines below if not using
# MPI
#######################
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
#######################

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

number_of_nodes = 1
miindmodel = miind.MiindModel(number_of_nodes)

miindmodel.init([])

timestep = miindmodel.getTimeStep()
print('Timestep from XML : {}'.format(timestep))

simulation_length = miindmodel.getSimulationLength() #s
print('Simulation Length (s) from XML : {}'.format(simulation_length))

# For MPI child processes, startSimulation runs the full simulation loop
# and so will not return until MPI process 0 has completed. At that point,
# we want to kill the child processes so that sim() is not called more than once.
if miindmodel.startSimulation() > 0 :
    quit()

# For Agonist/Antagonist relationship only
# prop_input = [a,b,c,d,0,0,0,0,0]
# bg_input = [500,500,500,500,310,310,310,310,310]
#  0 degrees : a = 0 ; b,c,d = 200
# 90 degrees : a = 0 ; b,c,d = 50

# For RF Flexor Bias
# prop_input = [a,b,c,d,0,0,0,0,0]
# bg_input = [500,500,500,0,310,310,310,310,310]
# To ignore Ag/Antag, a,b,c = 0
# Change d to alter the amount of RF bias
#  0 degrees : d = 450
# 90 degrees : d = 0

# For ST Extensor Bias
# prop_input = [a,b,c,d,0,0,0,0,0]
# bg_input = [500,500,0,500,310,310,310,310,310]
# To ignore Ag/Antag, a,b,d = 0
# Change d to alter the amount of ST bias
#  0 degrees : c = 400
# 90 degrees : c = 0

# Mix of everything
# prop_input = [a,b,c,d,0,0,0,0,0]
# bg_input = [500,500,0,0,310,310,310,310,310]
#  0 degrees : a = 0; b = 200; c = 500; d = 250
# 90 degrees : a = 0; b = 50 ; c = 500; d = 0

prop_input = [0,50,500,0,0,0,0,0,0]
bg_input = [500,500,0,0,310,310,310,310,310]
# each MN and assicuated INT gets the same supraspinal input
supra_input = [0,0,0,0,0,0,0,0,0]

outputs = []
t = 0.0

rate = [0.0,0.0,0.0,0.0,80.0,80.0,80.0,80.0,80.0]
start_flexion = (numpy.ones(len(supra_input))*3).tolist()
up_ramp = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
down_ramp = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
end_flexion = (numpy.ones(len(supra_input))*8).tolist()
# Differences in the maximum rate of each supraspinal input indicates which
# which muscles are the agonists + variation among muscle activation
#rate = [0,0,8000,8500,9000,9500,10000]
# rate_var = [0,0,100,-150,200,-250,300]
# rate = list( map(add, rate, rate_var) )
#rate = list( map(add, rate, prop_input) )

for z in range(int(simulation_length/timestep)):
    t += timestep

    for i in range(len(supra_input)):
        if(t > start_flexion[i]):
            if(t > end_flexion[i]):
                if(t < end_flexion[i]+down_ramp[i]):
                    supra_input[i] = (1.0-((t - end_flexion[i]) / (down_ramp[i]) ) ) * rate[i]
                else:
                    supra_input[i] = 0
            else:
                if(t < start_flexion[i]+up_ramp[i]):
                    supra_input[i] = ((t - start_flexion[i]) / (up_ramp[i]) ) * rate[i]
                else:
                    supra_input[i] = rate[i]
        else:
            supra_input[i] = 0

    node_input = list( map(add, supra_input, prop_input) )
    node_input = numpy.array(list( map(add, node_input, bg_input)))
    o = miindmodel.evolveSingleStep(node_input.tolist())
    if(t > 0.5):
        outputs.append(o)

miindmodel.endSimulation()

res_list = numpy.matrix(outputs)
res_list = numpy.transpose(res_list)

for i in range(5):
    temp = res_list[i].tolist()[0]
    temp = numpy.convolve(temp, numpy.ones(10000)/10000, mode='same')
    res_list[i][0] = numpy.transpose(temp)

bwah = res_list[:,10000:100000:5]

plt.figure()
plt.subplot(511)
plt.plot((bwah[0].tolist())[0])
plt.title("Firing Rates")
plt.subplot(512)
plt.plot((bwah[1].tolist())[0])
plt.subplot(513)
plt.plot((bwah[2].tolist())[0])
plt.subplot(514)
plt.plot((bwah[3].tolist())[0])
plt.subplot(515)
plt.plot((bwah[4].tolist())[0])

plt.show()

# normalise values per muscle
for i in range(5):
    bwah[i] += numpy.random.normal(0,1,bwah[i].shape)*0
    xmax, xmin = bwah[i].max(), bwah[i].min()
    bwah[i] = bwah[i]/xmax

# res_list = numpy.append(res_list, numpy.matrix([numpy.zeros(50),numpy.zeros(50),numpy.zeros(50),numpy.zeros(50),numpy.zeros(50)]), axis=1)

plt.figure()
plt.subplot(511)
plt.plot((bwah[0].tolist())[0])
plt.title("Firing Rates")
plt.subplot(512)
plt.plot((bwah[1].tolist())[0])
plt.subplot(513)
plt.plot((bwah[2].tolist())[0])
plt.subplot(514)
plt.plot((bwah[3].tolist())[0])
plt.subplot(515)
plt.plot((bwah[4].tolist())[0])
plt.plot((bwah[0].tolist())[0])

plt.show()

bwah = numpy.absolute(bwah)

comps = 2

nmf = nimfa.Nmf(bwah, seed="nndsvd", rank=comps, max_iter=500)
nmf_fit = nmf()

W = nmf_fit.basis().transpose()
H = nmf_fit.coef().transpose()

V = numpy.matmul(H,W)

plt.style.use('seaborn')

plt.figure()
axes = plt.gca()
axes.set_ylim([0,6.0])
plt.bar(['RF','VL','VM','ST','BF'],W[0,:].tolist()[0])
plt.title("NMF Coefficient " + str(0+1) + " (400Hz / 4nA)")
plt.show()

plt.figure()
axes = plt.gca()
axes.set_ylim([0,5.0])
plt.bar(['RF','VL','VM','ST','BF'],W[1,:].tolist()[0])
plt.title("NMF Coefficient " + str(1+1) + " (400Hz / 4nA)")
plt.show()

plt.figure()
plt.xlabel('Time (ms)')
axes = plt.gca()
axes.set_ylim([0,0.2])
plt.plot(H[:,0])
plt.title("NMF Factor " + str(1) + " (400Hz / 4nA)")
plt.show()

plt.figure()
plt.xlabel('Time (ms)')
axes = plt.gca()
axes.set_ylim([0,0.12])
plt.plot(H[:,1])
plt.title("NMF Factor " + str(2) + " (400Hz / 4nA)")
plt.show()
