# Once the Python Shared Library has been built in MIIND,
# copy this file to the results directory (where the .cpp and .so files were
# created).

import pylab
import numpy
import matplotlib
matplotlib.use('TkAgg')
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
# bg_input = [310,310,310,310,310,310,310,310,310]
#  0 degrees : prop_input = [190,440,440,440,0,0,0,0,0]
# 90 degrees : prop_input = [190,240,240,240,0,0,0,0,0]

# For RF Flexor Bias
# bg_input = [310,310,310,310,310,310,310,310,310]
#  0 degrees : prop_input = [190,190,190,0,0,0,0,0,0]
# 90 degrees : prop_input = [190,190,190,170,0,0,0,0,0]

# For ST Extensor Bias
# bg_input = [310,310,310,310,310,310,310,310,310]
#  0 degrees : prop_input = [190,190,0,190,0,0,0,0,0]
# 90 degrees : prop_input = [190,190,170,190,0,0,0,0,0]

# Mix of everything
# bg_input = [310,310,310,310,310,310,310,310,310]
#  0 degrees : prop_input = [190,390,190,0,0,0,0,0,0]
# 90 degrees : prop_input = [190,240,190,0,0,0,0,0,0]

prop_input = [190,390,190,0,0,0,0,0,0]
bg_input = [310,310,310,310,310,310,310,310,310]
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

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=(8,5))
fig.tight_layout()
fig.subplots_adjust(top=0.85)
fig.set_size_inches(18, 4.5)

#plt.figtext(0.25,0.93,"Synergy One", va="center", ha="center", size=30)
#plt.figtext(0.75,0.93,"Synergy Two", va="center", ha="center", size=30)

show_muscle_labels=False
show_column_title=False
col='#6666AA'

######
ax1.set_ylim([0,7.0])
rects1 = ax1.bar(['RF','VL','VM','ST','BF'], W[0,:].tolist()[0], 0.6, color=col, label='Position 1',capsize=2, alpha=1.0)

ax1.tick_params(axis='both',which='both',left=True,bottom=show_muscle_labels,labelbottom=show_muscle_labels,labelleft=True,labelsize=20)
ax1.yaxis.set_ticks(numpy.arange(0.0, 7.01, 1.0))
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.plot()

#######
ax2.set_ylim([0,0.2])
rects1 = ax2.plot([x * 0.0005 for x in range(len(H[:,0]))],H[:,0],color=col)

ax2.xaxis.set_ticks(numpy.arange(0, 9.0, 2.5))
ax2.tick_params(axis='both',which='both',left=True,bottom=show_muscle_labels,labelbottom=show_muscle_labels,labelleft=True,labelsize=20)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.plot()

#######
ax3.set_ylim([0,6.0])
rects1 = ax3.bar(['RF','VL','VM','ST','BF'], W[1,:].tolist()[0], 0.6, color=col, label='Position 1',capsize=2, alpha=1.0)

ax3.yaxis.set_ticks(numpy.arange(0.0, 6.01, 1.0))
ax3.tick_params(axis='both',which='both',left=True,bottom=show_muscle_labels,labelbottom=show_muscle_labels,labelleft=True,labelsize=20)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.plot()

#######
ax4.set_ylim([0,0.1])
rects1 = ax4.plot([x * 0.0005 for x in range(len(H[:,1]))],H[:,1],color=col)

ax4.xaxis.set_ticks(numpy.arange(0, 9.0, 2.5))
ax4.tick_params(axis='both',which='both',left=True,bottom=show_muscle_labels,labelbottom=show_muscle_labels,labelleft=True,labelsize=20)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.plot()

fig.savefig('avg_0.svg', dpi=100, format='svg')
fig.savefig('avg_0.png', dpi=100, format='png')

plt.show()
