#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:54:01 2019

@author: gareth
"""
import numpy
import imp
import sys
import csv
from operator import add
from random import randint
from mpi4py import MPI

def simulation_run(flex_rate=400.0,ext_rate=400.0):
    import libisopy as miind
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
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
    prop_input = [90,110,110,90,0,0,0,0,0]
    bg_input = [400,400,300,300,300,300,300,300,300]
    # each MN and assicuated INT gets the same supraspinal input
    supra_input = [0,0,0,0,0,0,0,0,0]
    outputs = []
    t = 0.0
    rate = [flex_rate,ext_rate,0.0,0.0,15.0,0.0,0.0,0.0,0.0]
    start_flexion = (numpy.ones(len(supra_input))*2).tolist()
    up_ramp = [1,1,1,1,1,1,1,1,1]
    down_ramp = [1,1,1,1,1,1,1,1,1]
    end_flexion = (numpy.ones(len(supra_input))*7).tolist()
    # Differences in the maximum rate of each supraspinal input indicates which
    # which muscles are the agonists + variation among muscle activation
    #rate = [0,0,8000,8500,9000,9500,10000]
    # rate_var = [0,0,100,-150,200,-250,300]
    # rate = list( map(add, rate, rate_var) )
    #rate = list( map(add, rate, prop_input))
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
    bwah = np.transpose(bwah[0:5,:])
    # write bwah to a file with a unique name based on the flex_rate and ext_rate
    numpy.savetxt(('output_' + str(flex_rate) + '_' + str(ext_rate) + '.csv'),bwah,delimiter=',')

    
if __name__ == '__main__':
    args = sys.argv # should be ['simulator.py',flex_rate,ext_rate]
    simulation_run(float(args[1]), float(args[2]))
