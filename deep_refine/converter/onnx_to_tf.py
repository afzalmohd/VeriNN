#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 15:46:35 2021

@author: u1411251
"""
import onnx
from onnx_translator import *
import numpy as np
import os


def convert(input_file, output_file):
    onnx_model = onnx.load(input_file)
    onnx.checker.check_model(onnx_model)
    translator = ONNXTranslator(onnx_model, False)
    operations, resources = translator.translate()
    
    file = open(output_file, "w+")
    
    
    i=1
    while i<len(operations):
        op = ""
        if operations[i] == 'Gemm':
            weight = resources[i]['deeppoly'][0]
            bias = resources[i]['deeppoly'][1]
            op = 'Gemm'
            i += 1
        if i < len(operations):
            if operations[i] == 'Relu':
                op = 'ReLU'
            i += 1
        file.write(op+"\n")
        file.write(str(weight.tolist())+"\n")
        #np.save(file, weight)
        #file.write("\n")
        if i < len(operations):
            file.write(str(bias.tolist())+"\n")
        else:
            file.write(str(bias.tolist()))
        

parent_path = '/home/u1411251/Documents/Phd/tools/networks'
onnx_dir =  parent_path+'/onnx/acasxu'
tf_dir = parent_path+'/tf/acasxu'

for file in os.listdir(onnx_dir):
    #t1 = str(file).split('.')
    #t1 = t1[:len(t1)-1]
    file_without_extension = os.path.splitext(str(file))[0]
    tf_file_path = tf_dir+'/'+file_without_extension+'.tf'
    onnx_file_path = onnx_dir+'/'+str(file)
    print(str(file))
    convert(onnx_file_path, tf_file_path)
    
            
