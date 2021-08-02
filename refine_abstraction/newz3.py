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
from PIL import Image
import create_actual_image
tf.disable_v2_behavior()

def read_n():
        pass

#Normalization of image
def normalize(image, means, stds):
        for i in range(len(image)):
                image[i] = (image[i] - means)/stds


"""Read csv file to get all images
present in dataset"""
def get_data(dataset):
        csvfile = open('{}_test.csv'.format(dataset), 'r')
        tests = csv.reader(csvfile, delimiter=',')
        return tests


#generate new image using eta values we get from model              
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
                #pixel values are appended
                newimage.append(np.float64(newdiff))
        return newimage

# ----------main function is present here-----------#
if __name__ == "__main__" :
        argumentList = sys.argv
        netname = argumentList[1]
        filename, file_extension = os.path.splitext(netname)
        epsilon = 0.05
        number_of_layers = 2
        dataset=argumentList[2]
        output_cons = argumentList[3]
        internal_cons = argumentList[4]
        tests = get_data(dataset)
        
        """Below is loop over all images in dataset but for now 
        we are using only first image. deepzono fails to verify 
        this image on epision 0.05"""
        
        # choose the first image
        test = ''
        for i, t in enumerate(tests):
                if i == 0:
                        test = t

        """image format : index 0 contains correct label
        and index 1 to length of array contains pixel values"""
        lbl = test[0]
        print("correct label: " + lbl)

        #image after normalizing
        original_image_pixels = np.array(np.float64(test[1:len(test)])/np.float64(255))
        
        #image without normalizing
        original_image2_pixels = np.array(np.float64(test[1:len(test)]))
        
        image_size = len(original_image2_pixels)
        
        """Below line helps to read network line by line and it returns
        1 model
        2 inp: tensorflow object where we can feed our image to get output using above model
        3 ls_obj: list of objects correcsponding to each nodes it helps to print all nodes values"""
        model, inp, means, stds, ls_obj = read_tensorflow_net(netname, image_size, True)#N

        #create image from pixels values
        create_actual_image.original(original_image2_pixels)

        
        #below is just to avoid divide by 0
        if stds != 0:
                normalize(original_image_pixels, means, stds) 
                        
        with tf.Session() as sess:
                pred = sess.run(model, feed_dict={inp: original_image_pixels})
                newpred = pred.flatten()
                        
        print("label of original image classified by model:"  + str(np.argmax(newpred)))
                        
        
        """Here we pass actual label of image to sat solver
        which solve some relevant constraints"""
        eta_set = set()
        s = Solver()
        modl = z3_format_converter.solve_cons(s, eta_set, output_cons, lbl, epsilon, original_image_pixels)
        eta_size = len(eta_set) 
        newimage = generate_new_image(lbl, test, modl)
        create_actual_image.generated(newimage)

        """Here ls_val at the end will have shape of [50, 50, 10] 
        corresponding to all layers in neural network
        It stores all intermediate as well as output layer
        nodes values"""
        ls_val=[]
        for x in ls_obj:
                with tf.Session() as sess:
                        pred = sess.run(x, feed_dict={inp: newimage})
                        newpred = pred.flatten()
                ls_val.append(newpred)
                        
        #save the image
        with open("testn.txt", "wb") as fp:
                pickle.dump(newimage, fp)
                        
        with open("testn.txt", "rb") as fp:
                b = pickle.load(fp)
                image = np.float64(b)

        if stds != 0:
                normalize(image, means, stds)   
                        
        c_maximum = np.argmax(newpred)
        
        print("label of generated image classified by model:" + str(c_maximum))
        
        maxsa = Solver()
        initial(modl, maxsa, image_size)
        if maxsa.check() == sat: 
                inter_modl = maxsa.model()
        else:
                print("unsat and can't proceed now")
                exit()
        eta_set = set()
        for x in range(0, image_size):
                eta_set.add(x)
        
        """keep track of all layer sizes in an array
        (exclude input layer and output layer)"""
        layerwise_size = []
        for i in range(0, number_of_layers):
                x = len(ls_val[i])
                layerwise_size.append(int(x))

        """solve inner layer constraints it will give eta_dd values 
        format:
        solve_cons_inner(solver,array of layer sizes,values of each nodes,internal_cons,etas)
        """
        solve_cons_inner(maxsa, layerwise_size, ls_val, internal_cons, eta_set)
        
        print(maxsa.check())
        
        if maxsa.check() == sat: 
                inter_modl = maxsa.model()
        else:
                print("unsat and can't proceed now")
                exit()
        
        """store eta_dd values we get from inter_model in a dict
        and print which are out of range(just for debugging)"""
        
        eta_dd = {}
        print("values outside range...")
        for x in eta_set:
                t = 'eps' + str(x)
                u = t  + 'dd'
                key = u
                u = Real(u)
                res = inter_modl[u]
                value = float(res.numerator_as_long())/float(res.denominator_as_long())
                if value > 1 or value < -1:
                        print(str(value) + ' ' + str(key))
                eta_dd[key] = value
        print("------------------------")
        
        """after getting eta_dd values solve 3 more constraints
        1. label should not be correct
        2. soft constraints for each node : a bool varibale (t)
        3. t -> (eta == eta_dd)"""
        newSolver = Optimize()
        eta_set = set()
        for x in range(0, image_size):
                eta_set.add(x)
                var = 'eps' + str(x)
                newvar = var + 'dd'
                var = Real(var)
                newSolver.add(var == eta_dd[newvar])
        mod = solve_cons_out(newSolver, layerwise_size,eta_set, eta_dd, lbl, output_cons,internal_cons)

for x in range(0, 50):
                t = "(" + str(x) + ")"+ "_0_b"
                u = "(" + str(x) + ")" + "_1_b"
                print(t)
                t = Bool(t)
                u = Bool(u)
                print(str(mod[t]) + " " + str(mod[u]))
# ------- eta values-------------------#
# print(eta_dd)
# for x in eta_set:
#         t = 'eps' + str(x)
#         t = Real(t)
#         res = mod[t]
#         value = float(res.numerator_as_long())/float(res.denominator_as_long())
#         print(str(x) + ':' + str(value))