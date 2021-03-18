#!/usr/bin/python3
import os
from z3 import *
from read_net_file import *
import sys
import csv
import pickle
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.python.keras import backend as K
import modifier
import z3_format_converter
import numpy as np
import eta_d_dash
from maxsat import *
tf.disable_v2_behavior()
def read_n():
        pass
'''

Normalization of image

'''

def normalize(image, means, stds):
        for i in range(len(image)):
                image[i] = (image[i] - means)/stds


'''

Read csv file to get all images
present in dataset

'''

def get_data(dataset):
        csvfile = open('{}_test.csv'.format(dataset), 'r')
        tests = csv.reader(csvfile, delimiter=',')
        return tests

'''

generate new image using eta values we get 
from model              

'''

def generate_new_image(lbl,test,modl):
        image= np.float64(test[1:len(test)])/np.float64(255)
        epsilon = 0.05
        #lower bound of image
        specLB = np.clip(image - epsilon,0,1)
        #upper bound of image
        specUB = np.clip(image + epsilon,0,1)
        newimage = []
        for j in range(0, 784):
                t = 'eps' + str(j)
                t = Real(t)
                res = modl[t]
                x = float(res.numerator_as_long())/float(res.denominator_as_long())
                diff = np.float64(specUB[j]) - np.float64(specLB[j])
                newdiff = np.float64(image[j]) + epsilon*x

                newimage.append(np.float64(newdiff))
        return newimage

'''

main function is present here

'''

if __name__ == "__main__" :
        argumentList = sys.argv
        netname = argumentList[1]
        filename, file_extension = os.path.splitext(netname)
        epsilon = 0.05
        '''
        
        Below line helps to read network line by line and 
        returns 

        model
        inp : tensorflow object where we can feed our image 
        to get output using above model.
        ls_obj : list of objects correcsponding to each layers
        it helps to print all nodes values. 

        '''
        
        model, inp, means, stds, ls_obj = read_tensorflow_net(netname, 784, True)#N
        
        dataset=argumentList[2]
        output_cons = argumentList[3]
        internal_cons = argumentList[4]
        tests = get_data(dataset)
        
        '''
        Below is loop over all images in dataset
        but fo now we are using only first image
        deepzono fails to verify this image using 
        on epision 0.03 

        '''

        # choose the image
        test = ''
        for i, t in enumerate(tests):
                if i == 0:
                        test = t
        
        # for i, test in enumerate(tests):
        #         if i == 0:
        print("correct label:" +test[0])
        lbl = test[0]
                        
        newtest= np.array(np.float64(test[1:len(test)])/np.float64(255))

        '''
        
        it is just to avoid divide by 0
        
        '''
        # why?
        if stds != 0:
                normalize(newtest, means, stds) 
        
                        
        '''
        
        Here ls_val has shape of [50, 50, 10] 
        corresponding to all layers in neural network
        It stores all intermediate as well as output layer
        nodes values

        '''
        ls_val=[]
        for x in ls_obj:
                with tf.Session() as sess:
                        pred = sess.run(x, feed_dict={inp: newtest})
                        newpred = pred.flatten()
                ls_val.append(newpred)
                        
                        
        maximum = np.argmax(newpred)
                        
        print("classified label:" + str(maximum))
        
        '''
        
        Here we pass actual label of image to sat solver
        which solve some relevant constraints
        
        '''
        
        modl = z3_format_converter.solve_cons(output_cons, lbl, epsilon, newtest)
        #print(modl)
        #exit()
        #break
        newimage = generate_new_image(lbl, test, modl)
        print(newimage)
        '''
        
        save the image 

        '''
        with open("testn.txt", "wb") as fp:
                pickle.dump(newimage, fp)
                        
        with open("testn.txt", "rb") as fp:
                b = pickle.load(fp)
                image= np.float64(b)
        if stds!= 0:
                normalize(image, means, stds)   
                        
        with tf.Session() as sess:
                c_pred = sess.run(model,feed_dict={inp:image})

        #all_nodes = [n for n in tf.get_default_graph().as_graph_def().node]
        #print(all_nodes)

        #print(c_pred)
        #print(pred)
                        
        c_newpred = c_pred.flatten()
                        
        c_maximum = np.argmax(c_newpred)

        print(c_maximum)
        
        maxsa = Solver()
        initial(modl, maxsa, 784)
        #print(maxsa.check())
        #print(maxsa.model())
        #exit()
        #break
        solve_cons_inner(maxsa,len(ls_val[1]), len(ls_val[2]), ls_val, internal_cons)
        
        mod = solve_cons_out(maxsa, ls_val, modl, len(ls_val[0]),870,lbl, output_cons)
        print(mod)
        exit()
        read_n()
