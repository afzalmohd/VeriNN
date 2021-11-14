#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:09:35 2021

@author: u1411251
"""

import os

ERANROOT = "/home/u1411251/Documents/Phd/tools/eran"
ERANMAIN = ERANROOT+"/tf_verify"
DREFINEROOT = "/home/u1411251/Documents/Phd/tools/VeriNN/deep_refine"
OUTPATH = DREFINEROOT+"/outfiles"
DEEPPOLYOUT = OUTPATH+"/deeppoly_out.txt"
OUTPUTBU = OUTPATH+"/marked_bounds_updated.txt"
OUTASSIGN = OUTPATH+"/assign.txt"
DREFINEOUT = OUTPATH+"/marked_neurons.txt"
NETPATH = DREFINEROOT+"/benchmarks/networks/mnist_relu_3_50.tf"
DATASETFILE = DREFINEROOT+"/benchmarks/dataset/mnist/mnist_test.csv"

epsilon = 0.04
dataset = "mnist"
domain = "deeppoly"
k = 4


if not os.path.isdir(OUTPATH):
    os.mkdir(OUTPATH)
else:
    if os.path.isfile(DREFINEOUT):
        os.remove(DREFINEOUT)
    if os.path.isfile(OUTPUTBU):
        os.remove(OUTPUTBU)


os.environ['OUTPUTBU'] = OUTPUTBU
os.environ['DREFINEOUT'] = DREFINEOUT
os.environ['DEEPPOLYOUT'] = DEEPPOLYOUT
os.environ['OUTASSIGN'] = OUTASSIGN
os.environ['ISDREFINE'] = "Y"
os.chdir(ERANMAIN)
print("In directory: "+os.getcwd())

print("-----------------------KPOLY STARTED-----------------------------")

#os.system("python3 . --netname "+NETPATH+" --epsilon "+str(epsilon)+" --domain "+domain+" --dataset "+dataset)

domain = "refinepoly"
os.system("python3 . --netname "+NETPATH+" --epsilon "+str(epsilon)+" --domain "+domain+" --dataset "+dataset+" --k "+str(k))                   

counter = 0
counter_limit  = 10

while counter < counter_limit:
    counter += 1
    os.chdir(DREFINEROOT)

    print("------------------------DREFINE STARTED------------------------------")
    os.system("./drefine -f "+DEEPPOLYOUT+" --network "+NETPATH+" --dataset-file "+DREFINEROOT+" --epsilon "+str(epsilon)+" --m "+DREFINEOUT)


    print("-----------------------KPOLY STARTED---------------------------------")
    os.chdir(ERANMAIN)  
    domain = "refinepoly"
    os.system("python3 . --netname "+NETPATH+" --epsilon "+str(epsilon)+" --domain "+domain+" --dataset "+dataset+" --k "+str(k))     

    if os.environ.get('ISTERMINATE') == "Y":
        print("Terminated after {} iterations".format(counter))    
        break;                 
    
    #print("-----------------------DEEPPOLY STARTED-----------------------------")
    #domain = "deeppoly"
    #os.system("python3 . --netname "+NETPATH+" --epsilon "+str(epsilon)+" --domain "+domain+" --dataset "+dataset)







