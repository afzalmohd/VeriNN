#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:09:35 2021

@author: u1411251
"""

from genericpath import isdir, isfile
from multiprocessing import Pool
import os
import csv
import sys
import time
import random

NUM_CPU = 7
TIMEOUT = 2000
DATASET = "MNIST"
NUM_IMAGES = 100
num_cores = 1
tool_name = "drefine" #drefine

root_dir = os.getcwd()
TOOL = os.path.join(root_dir, 'drefine')
result_dir = os.path.join(root_dir, 'outfiles')
dataset_file = os.path.join(root_dir, 'benchmarks/dataset/mnist/mnist_test.csv')
result_file = os.path.join(result_dir, "dumb_bounds.txt")

if not os.path.isdir(result_dir):
    os.mkdir(result_dir)


def write_script_file(file_name, cmds):
    with open(file_name, 'w') as file:
        for cm in cmds:
            file.write(cm+"\n")
        file.close()

def get_tasks_from_file():
    file_path = '/home/afzal/Documents/tools/VeriNN/deep_refine/diff_ab_drefine.txt'
    f = open(file_path, 'r')
    Lines = f.readlines()
    rows = []
    for line in Lines:
        line = line.strip()
        line = line.split(',')
        net = line[0]
        net1 = net[:-5]
        prp = line[1][:-7]
        rows.append(net1+"+"+prp)

    return rows


def get_bounds_tasks():
    bounds_dir = '/home/afzal/tools/alpha-beta-CROWN/dumb_bounds'
    net_dir = '/home/afzal/tools/networks/tf/mnist'
    prp_dir = '/home/afzal/tools/networks/props/mnist'
    tasks = []
    for file_name in os.listdir(bounds_dir):
        file_path_s = file_name.split('+')
        net_name = file_path_s[0]+".tf"
        prp_name = file_path_s[1]+".vnnlib"
        prp_dir1 = os.path.join(prp_dir, file_path_s[0].replace('_','-'))
        time_taken = float(file_path_s[2])
        remaining_time = TIMEOUT - time_taken
        file_path = os.path.join(bounds_dir, file_name)
        tasks.append([os.path.join(net_dir, net_name), os.path.join(prp_dir1, prp_name), file_path, remaining_time])

    return tasks

def filtered_tasks():
    tasks = get_bounds_tasks()
    file_tasks = get_tasks_from_file()
    tasks1 = []

    for fts in file_tasks:
        for t in tasks:
            if fts in t[2]:
                tasks1.append(t)
                break

    return tasks1






def run_command_bounds(net_path, prp_path, bounds_path, timeout, cpu_idx):
    net_name = os.path.basename(net_path)[:-3]
    result_file = os.path.join(result_dir, f"bounds_res.txt")
    command = f"taskset --cpu-list {cpu_idx}-{cpu_idx} timeout -k 2s {timeout} {TOOL} --network {net_path} --vnnlib-prp-file {prp_path} --bounds-path {bounds_path} --dataset {DATASET} --result-file {result_file}"
    print(command)
    os.system(command)

def run_per_cpu_bounds(tasks_per_cpu):
    for task in tasks_per_cpu:
        run_command_bounds(task[0], task[1], task[2], task[3], task[4])

def print_cmnds_bounds(num_cpu, log_dir):
    tasks = get_bounds_tasks()
    # tasks = filtered_tasks()
    # tasks = tasks[:70]

    random.shuffle(tasks)

    num_tasks = len(tasks)
    print(f"Total number of task: {num_tasks}")

    if num_cpu >= num_tasks:
        load_per_cpu = [1]*num_tasks
    else:
        load_per_cpu = [0]*num_cpu
        for i in range(0,num_tasks):
            j = i % num_cpu
            load_per_cpu[j] += 1

    print("Load per cpu: {}".format(load_per_cpu))

    prev_load = 0
    for idx, load in enumerate(load_per_cpu):
        ld = tasks[prev_load:prev_load+load]
        prev_load += load
        cmds = []
        for l in ld:
            net_path = l[0]
            prp_path = l[1]
            bounds_path = l[2]
            timeout = l[3]
            net_name = os.path.basename(net_path)
            prp_name = os.path.basename(prp_path)
            log_file = net_name+"+"+prp_name
            log_file = os.path.join(log_dir, log_file)
            result_file = os.path.join(result_dir, f"file_{idx}.txt")
            command = f"taskset --cpu-list {10*idx}-{(10*idx) + 9} timeout -k 2s {timeout} {TOOL} --network {net_path} --vnnlib-prp-file {prp_path} --bounds-path {bounds_path} --dataset {DATASET} --result-file {result_file} >> {log_file}"
            cmds.append(command)
        file_name = os.path.join(log_dir, f"script_{idx}.sh")
        write_script_file(file_name, cmds)

    









def run_command(net_path, ep, image_index, cpu_idx):
    net_name = os.path.basename(net_path)[:-3]
    command = f"taskset --cpu-list {cpu_idx}-{cpu_idx} timeout -k 2s {TIMEOUT} {TOOL} --network {net_path} --dataset-file {dataset_file} --epsilon {ep} --dataset {DATASET} --result-file {result_file} --image-index {image_index}"
    print(command)
    os.system(command)

def run_per_cpu(tasks_per_cpu):
    for task in tasks_per_cpu:
        run_command(task[0], task[1], task[2], task[3])



def run_command_vnnlib(net_path, prp_path, cpu_idx):
    net_name = os.path.basename(net_path)[:-3]
    result_file = os.path.join(result_dir, f"vnnlib_res.txt")
    command = f"taskset --cpu-list {cpu_idx}-{cpu_idx} timeout -k 2s {TIMEOUT} {TOOL} --network {net_path} --vnnlib-prp-file {prp_path} --dataset {DATASET} --result-file {result_file}"
    print(command)
    os.system(command)

def run_per_cpu_vnnlib(tasks_per_cpu):
    for task in tasks_per_cpu:
        run_command_vnnlib(task[0], task[1], task[2])


def get_tasks_vnnlib():
    net_dir = '/home/afzal/Documents/tools/networks/vnncomp21/mnist'
    prp_dir = net_dir
    nets = []
    prps = []
    for file_name in os.listdir(net_dir):
        file_name = os.path.join(net_dir, file_name)
        if file_name.endswith("tf"):
            nets.append(file_name)
        elif file_name.endswith("vnnlib"):
            prps.append(file_name)

    tasks = []
    for nt in nets:
        for prp in prps:
            tasks.append([nt, prp])
    
    return tasks

def run_with_vnnlib():
    tasks = get_tasks_vnnlib()
    NUM_TASK = len(tasks)
    print(f"Total number of task: {NUM_TASK}")

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
    for idx, load in enumerate(load_per_cpu):
        ld = tasks[prev_load:prev_load+load]
        for l in ld:
            l.append(idx)
        task_per_cpu.append(ld)
        prev_load += load

    # print(task_per_cpu)
    with Pool(processes=len(task_per_cpu)) as p:
        p.map(run_per_cpu_vnnlib, task_per_cpu)


def run_with_bounds():
    tasks = get_bounds_tasks()
    NUM_TASK = len(tasks)
    print(f"Total number of task: {NUM_TASK}")

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
    for idx, load in enumerate(load_per_cpu):
        ld = tasks[prev_load:prev_load+load]
        for l in ld:
            l.append(idx)
        task_per_cpu.append(ld)
        prev_load += load

    # print(task_per_cpu)
    with Pool(processes=len(task_per_cpu)) as p:
        p.map(run_per_cpu_bounds, task_per_cpu)

def get_task_from_file_random():
    file_name = "/home/afzal/tools/alpha-beta-CROWN/random_bench_1.csv"
    file1 = open(file_name, 'r')
    Lines = file1.readlines()
    tasks = []
    for line in Lines:
        line = line.strip()
        list_line = line.split(",")
        net_name = list_line[0][:-5]+".tf"
        prop_list = list_line[1][:-7].split("_")
        image_index = int(prop_list[1])
        ep = float(prop_list[2])
        tasks.append([net_name, ep, image_index])

    return tasks


def get_all_tasks():
    tasks = []
    # net_dir = '/home/afzal/Documents/tools/networks/tf/mnist'
    NETWORK_FILE = []
    NETWORK_FILE = ["mnist_relu_3_50.tf","mnist_relu_3_100.tf"]
    NETWORK_FILE += ["mnist_relu_6_100.tf", "mnist_relu_5_100.tf", "mnist_relu_6_200.tf"]
    # NETWORK_FILE += ["mnist_relu_4_1024.tf"]
    NETWORK_FILE += ["mnist_relu_9_100.tf", "mnist_relu_9_200.tf"]
    # NETWORK_FILE += ["ffnnRELU__Point_6_500.tf", "ffnnRELU__PGDK_w_0.1_6_500.tf", "ffnnRELU__PGDK_w_0.3_6_500.tf"]

    epsilons = [0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05]

    for image_index in range(NUM_IMAGES):
        for ep in epsilons:
            for nt in NETWORK_FILE:
                tasks.append([nt, ep, image_index])

    return tasks

def print_cmnds_all(num_cpu, log_dir):
    tasks = get_all_tasks()
    # tasks = get_task_from_file_random()
    net_dir = '/home/afzal/tools/networks/tf/mnist'
    random.shuffle(tasks)

    num_tasks = len(tasks)
    print(f"Total number of task: {num_tasks}")

    if num_cpu >= num_tasks:
        load_per_cpu = [1]*num_tasks
    else:
        load_per_cpu = [0]*num_cpu
        for i in range(0,num_tasks):
            j = i % num_cpu
            load_per_cpu[j] += 1

    print("Load per cpu: {}".format(load_per_cpu))

    prev_load = 0
    for idx, load in enumerate(load_per_cpu):
        ld = tasks[prev_load:prev_load+load]
        prev_load += load
        cmds = []
        for l in ld:
            net_name = l[0][:-3]
            ep = l[1]
            image_index = l[2]
            net_path = os.path.join(net_dir, net_name+".tf")
            log_file = net_name+"+"+str(image_index)+"+"+str(ep)
            log_file = os.path.join(log_dir, log_file)
            result_file = os.path.join(result_dir, f"file_{idx}.txt")
            command = f"taskset --cpu-list {num_cores*idx}-{(num_cores*idx)+(num_cores -1)} timeout -k 2s {TIMEOUT} {TOOL} --tool {tool_name} --network {net_path} --dataset-file {dataset_file} --image-index {image_index} --epsilon {ep} --dataset {DATASET} --result-file {result_file} >> {log_file}"
            cmds.append(command)
        file_name = os.path.join(log_dir, f"script_{idx}.sh")
        write_script_file(file_name, cmds)







if __name__ == '__main__':
    if len(sys.argv) == 3:
        num_cpu = int(sys.argv[1])
        log_dir = sys.argv[2]
    else:
        print("Error: ")
        sys.exit(1)

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    print_cmnds_all(num_cpu, log_dir)  

    exit(0)

    tasks = get_all_tasks()
    NUM_TASK = len(tasks)
    print(f"Total number of task: {NUM_TASK}")

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
    for idx, load in enumerate(load_per_cpu):
        ld = tasks[prev_load:prev_load+load]
        for l in ld:
            l.append(idx)
        task_per_cpu.append(ld)
        prev_load += load


    with Pool(processes=len(task_per_cpu)) as p:
        p.map(run_per_cpu, task_per_cpu)








