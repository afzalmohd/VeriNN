#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:09:35 2021

@author: u1411251
"""

import os
from datetime import datetime

NUMTEST = 100
epsilons = [0.03,0.04,0.05]
epsilons = [0.03]
NETWORK_FILE = "ffnnRELU__PGDK_w_0.3_6_500.tf"
IS_MILP_REFINE = 0
IS_OPTIMIZATION_MARK = 0
drefine_timeout = 1000
DREFINEROOT = "/home/u1411251/Documents/Phd/tools/VeriNN/deep_refine"
OUTPATH = DREFINEROOT+"/outfiles"
NETPATH = "/home/u1411251/Documents/Phd/tools/networks/tf/mnist/"+NETWORK_FILE
NETPATH = "/home/u1411251/Documents/Phd/tools/networks/my_converter/mnist/"+NETWORK_FILE
DATASETFILE = DREFINEROOT+"/benchmarks/dataset/mnist/mnist_test.csv"
SCRIPT_OUT = "script_out.txt"
dataset = "mnist"

APPROACH = ""
if IS_MILP_REFINE == 1 and IS_OPTIMIZATION_MARK == 1:
    APPROACH = "MILP_WITH_MILP"
elif IS_MILP_REFINE == 0 and IS_OPTIMIZATION_MARK == 0:
    APPROACH = "PATHSPLIT_WITH_PULLBACK"
else:
    print("Wrong options\n")
    exit(0)


def write_to_file(image_index, epsilon):
    my_file = open(SCRIPT_OUT, "a")
    now = datetime.now()
    time = now.strftime("%H:%M")
    my_file.write(NETWORK_FILE+" , "+str(epsilon)+" , "+APPROACH+" , "+str(image_index)+" , "+time+"\n")
    my_file.close()

for epsilon in epsilons:
    RESULT_FILE = OUTPATH+"/"+NETWORK_FILE[:-3]+"_"+APPROACH+"_"+str(epsilon)+".txt"
    for i in range(1,NUMTEST+1):
        #write_to_file(i, epsilon)
        command = "timeout -k 2s "+str(drefine_timeout)+" ./drefine --network "+NETPATH+" --epsilon "+str(epsilon)+" --dataset-file "+DATASETFILE+" --result-file "+RESULT_FILE+" --is-milp-refine "+str(IS_MILP_REFINE)+" --is-optimization-mark "+str(IS_OPTIMIZATION_MARK)+" --image-index "+str(i)+" --tool deeppoly --is-parallel 1"                                   
        print(command)
        os.system(command)






