#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:09:35 2021

@author: u1411251
"""

from multiprocessing import Pool
import os
from datetime import datetime

NUM_CPU = 3
NUMTEST = 100
epsilons = [0.03,0.04,0.05]
epsilons = [0.03]
NETWORK_FILE = "ffnnRELU__PGDK_w_0.3_6_500.tf"
IS_MILP_REFINE = 1
IS_OPTIMIZATION_MARK = 1
TIMEOUT = 1000
DREFINEROOT = "/home/u1411251/Documents/Phd/tools/VeriNN/deep_refine"
OUTPATH = DREFINEROOT+"/outfiles"
NETPATH = "/home/u1411251/Documents/Phd/tools/networks/tf/mnist/"+NETWORK_FILE
NETPATH = "/home/u1411251/Documents/Phd/tools/networks/my_converter/mnist/"+NETWORK_FILE
DATASETFILE = DREFINEROOT+"/benchmarks/dataset/mnist/mnist_test.csv"
DREFINE_TOOL = "/home/u1411251/Documents/Phd/tools/VeriNN/deep_refine/drefine"
SCRIPT_OUT = "script_out.txt"
DATASET = "ACASXU"

APPROACH = ""
if IS_MILP_REFINE == 1 and IS_OPTIMIZATION_MARK == 1:
    APPROACH = "MILP_WITH_MILP"
elif IS_MILP_REFINE == 0 and IS_OPTIMIZATION_MARK == 0:
    APPROACH = "PATHSPLIT_WITH_PULLBACK"
else:
    print("Wrong options\n")
    exit(0)


# def write_to_file(image_index, epsilon):
#     my_file = open(SCRIPT_OUT, "a")
#     now = datetime.now()
#     time = now.strftime("%H:%M")
#     my_file.write(NETWORK_FILE+" , "+str(epsilon)+" , "+APPROACH+" , "+str(image_index)+" , "+time+"\n")
#     my_file.close()

# for epsilon in epsilons:
#     RESULT_FILE = OUTPATH+"/"+NETWORK_FILE[:-3]+"_"+APPROACH+"_"+str(epsilon)+".txt"
#     for i in range(1,NUMTEST+1):
#         #write_to_file(i, epsilon)
#         command = "timeout -k 2s "+str(drefine_timeout)+" ./drefine --network "+NETPATH+" --epsilon "+str(epsilon)+" --dataset-file "+DATASETFILE+" --result-file "+RESULT_FILE+" --is-milp-refine "+str(IS_MILP_REFINE)+" --is-optimization-mark "+str(IS_OPTIMIZATION_MARK)+" --image-index "+str(i)+" --tool deeppoly --is-parallel 1"                                   
#         print(command)
#         os.system(command)


def run_command(result_file, net_path, prp_file):
    epsilon = 0.03
    command = "timeout -k 2s "+str(TIMEOUT)+" "+DREFINE_TOOL+" --network "+net_path+" --epsilon "+str(epsilon)+" --dataset-file "+DATASETFILE+" --result-file "+result_file+" --is-milp-refine "+str(IS_MILP_REFINE)+" --is-optimization-mark "+str(IS_OPTIMIZATION_MARK)+" --vnnlib-prp-file "+prp_file+" --dataset "+DATASET
    print(command)
    os.system(command)

def run_per_cpu(tasks_per_cpu):
    for task in tasks_per_cpu:
        run_command(task[0], task[1], task[2])


net_path = "/home/u1411251/Documents/Phd/tools/networks/tf/acasxu"
prop_path = "/home/u1411251/Documents/Phd/tools/vnncomp2021/benchmarks/acasxu"
network_files = []
for i in range(1, 6):
    for j in range(1,10):
        net = net_path+"/ACASXU_run2a_{}_{}_batch_2000.tf".format(i,j)
        network_files.append(net)

props = []
for i in range(1,11):
    prp = prop_path+"/prop_{}.vnnlib".format(i)
    props.append(prp)

TASKS = []
result_file = "/home/u1411251/Documents/Phd/tools/VeriNN/deep_refine/outfiles/acasxu_results.txt"
for net in network_files:
    for prp in props:
        TASKS.append((result_file, net, prp))

NUM_TASK = len(TASKS)
print("Total number of task: {}".format(NUM_TASK))

if NUM_CPU >= NUM_TASK:
    load_per_cpu = [1]*NUM_TASK
else:
    load_per_cpu = [0]*NUM_CPU
    for i in range(0,NUM_TASK):
        j = i % NUM_CPU
        load_per_cpu[j] += 1

print("Load per cpu: {}".format(load_per_cpu))
task_per_cpu = []
prev_load = 0
for load in load_per_cpu:
    task_per_cpu.append(TASKS[prev_load:prev_load+load])
    prev_load += load

with Pool(processes=len(task_per_cpu)) as p:
    p.map(run_per_cpu, task_per_cpu)








