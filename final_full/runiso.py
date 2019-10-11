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
miind.init(number_of_nodes)

timestep = miind.getTimeStep()
print('Timestep from XML : {}'.format(timestep))

simulation_length = miind.getSimulationLength() #s
print('Simulation Length (s) from XML : {}'.format(simulation_length))

# For MPI child processes, startSimulation runs the full simulation loop
# and so will not return until MPI process 0 has completed. At that point,
# we want to kill the child processes so that sim() is not called more than once.
miind.startSimulation()

# pops = [Flex Aff,Ext Aff,InhibST,InhibRF,RF,Vl,VL,ST,BF]
# bg_input = [400,400,400,300,300,300,300,300,300]
#  0 degrees : prop_input = [90,180,100,10,0,0,0,0,0]
# 20 degrees : prop_input = [90,164,100,10,0,0,0,0,0]
# 30 degrees : prop_input = [90,156,100,10,0,0,0,0,0]
# 60 degrees : prop_input = [90,133,100,10,0,0,0,0,0]
# 90 degrees : prop_input = [90,110,100,10,0,0,0,0,0]

filename = 'avg_90'
prop_input = [90,180,100,10,0,0,0,0,0]
bg_input = [400,400,400,300,300,300,300,300,300]
# each MN and assicuated INT gets the same supraspinal input
supra_input = [0,0,0,0,0,0,0,0,0]

outputs = []
t = 0.0

rate = [60.0,60.0,0.0,0.0,45.0,20.0,20.0,20.0,20.0]
start_flexion = (numpy.ones(len(supra_input))*2).tolist()
up_ramp = [1,1,1,1,1,1,1,1,1]
down_ramp = [1,1,1,1,1,1,1,1,1]
end_flexion = (numpy.ones(len(supra_input))*7).tolist()
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
    o = miind.evolveSingleStep(node_input.tolist())
    outputs.append(o)

miind.endSimulation()

res_list = numpy.matrix(outputs)
res_list = numpy.transpose(res_list)

for i in range(5):
    temp = res_list[i].tolist()[0]
    temp = numpy.convolve(temp, numpy.ones(10000)/10000, mode='same')
    #temp = temp / numpy.max(temp)
    res_list[i][0] = numpy.transpose(temp)

bwah = res_list[:,10000:90000:20]
bwah = bwah[0:5,:]

numpy.savetxt(filename + '_raw.csv', numpy.transpose(bwah), delimiter=",")
print(bwah.shape)
plt.figure()
plt.subplot(511)
plt.plot((bwah[0].tolist())[0])
#plt.title("Firing Rates")
plt.subplot(512)
plt.plot((bwah[1].tolist())[0])
plt.subplot(513)
plt.plot((bwah[2].tolist())[0])
plt.subplot(514)
plt.plot((bwah[3].tolist())[0])
plt.subplot(515)
plt.plot((bwah[4].tolist())[0])

plt.show()

fig,ax = plt.subplots(5,1,figsize=(5,8))

for i in range(5):
	rects1 = ax[i].plot((bwah[i].tolist())[0],color='#000000', linewidth=4)
	ax[i].tick_params(axis='both',which='both',left=False,bottom=(i == 4),labelbottom=(i == 4),labelleft=True,labelsize=20)
	ax[i].spines["top"].set_visible(False)
	ax[i].spines["right"].set_visible(False)
	ax[i].spines["left"].set_visible(True)

fig.savefig(filename + '_raw_plot.svg', dpi=100, format='svg')
fig.savefig(filename + '_raw_plot.png', dpi=100, format='png')

# Save data to csv files for future use

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

numpy.savetxt(filename + '_coeff.csv', H, delimiter=",")
numpy.savetxt(filename + '_vector.csv', W.transpose(), delimiter=",")

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize=(5,8))
fig.tight_layout()
fig.subplots_adjust(top=0.95)
fig.set_size_inches(3, 18)

#plt.figtext(0.25,0.93,"Synergy One", va="center", ha="center", size=30)
#plt.figtext(0.75,0.93,"Synergy Two", va="center", ha="center", size=30)

show_muscle_labels=True
show_column_title=False
col='red'

######
ax1.set_ylim([0,5.0])
rects1 = ax1.bar(['RF','VL','VM','ST','BF'], W[0,:].tolist()[0], 0.6, color=col, label='Position 1',capsize=2, alpha=1.0, edgecolor='#000000',linewidth=2, hatch="//")
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
ax1.tick_params(axis='both',which='both',left=False,bottom=False,labelbottom=show_muscle_labels,labelleft=False,labelsize=20)
ax1.yaxis.set_ticks(numpy.arange(0.0, 5.01, 1.0))
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.plot()

#######
ax2.set_ylim([0,0.25])
rects1 = ax2.plot([x * 0.002 for x in range(len(H[:,0]))],H[:,0],color=col, linewidth=3)

ax2.xaxis.set_ticks(numpy.arange(0, 8.01, 8))
ax2.tick_params(axis='both',which='both',left=False,bottom=False,labelbottom=show_muscle_labels,labelleft=False,labelsize=20)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.plot()

#######
ax3.set_ylim([0,4.0])
rects1 = ax3.bar(['RF','VL','VM','ST','BF'], W[1,:].tolist()[0], 0.6, color=col, label='Position 1',capsize=2, alpha=1.0, edgecolor='#000000',linewidth=2, hatch="//")
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
ax3.yaxis.set_ticks(numpy.arange(0.0, 4.01, 1.0))
ax3.tick_params(axis='both',which='both',left=False,bottom=False,labelbottom=show_muscle_labels,labelleft=False,labelsize=20)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["left"].set_visible(False)
ax3.plot()

#######
ax4.set_ylim([0,0.13])
rects1 = ax4.plot([x * 0.002 for x in range(len(H[:,1]))],H[:,1],color=col, linewidth=3)

ax4.xaxis.set_ticks(numpy.arange(0, 8.01, 8))
ax4.tick_params(axis='both',which='both',left=False,bottom=False,labelbottom=show_muscle_labels,labelleft=False,labelsize=20)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.spines["left"].set_visible(False)
ax4.plot()

fig.savefig(filename + '.svg', dpi=100, format='svg')
fig.savefig(filename + '.png', dpi=100, format='png')

# Save data to csv files for future use


plt.show()
