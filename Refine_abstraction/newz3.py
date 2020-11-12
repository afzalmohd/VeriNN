from z3 import *
from read_net_file import *
import sys
import csv
import pickle
sys.path.insert(0, '../deepg/code/')
import tensorflow as tf
from tensorflow.python.keras import backend as K
import modifier
import z3_format_converter
import original_image
import numpy as np
sys.setrecursionlimit(1000000000)

def read_n():
	pass

def get_next_layer(d):

	imagee = original_image.get_image()
	Y = np.array(np.float64(imagee))
	print(Y)
	X = np.array(np.float64(imagee)/np.float64(255))

	padding = 1
	padded_image = np.zeros([X.shape[0] +2*padding, X.shape[1]+2*padding])
	padded_image[:X.shape[0],:X.shape[1]] = X
	
	#padded_image[1][1] = 1.0
	#padded_image[1][2] = 1.0
	'''for i in range(1,29):
		padded_image[1][i]=1.0
		padded_image[2][i]=1.0'''
	for j in range(1,25):
		for k in range(1,29):
			padded_image[j][k]= 1.0

	#--------------printting the padded image------------------#
	#print(padded_image)
	#print(padded_image.shape)

	kernal_x = len(d)
	kernal_y = len(d[0])
	stride = 2
	bias = original_image.get_bias()
	# 4 * 4 Kernal for multiplication
	ker = [[[0 for k in range(4)] for i in range(4)] for j in range(32)]
	kernal = np.array(ker, dtype = float)
	
	for i in range(len(d)):
		for j in range(len(d[0])):
			t = d[i][j][0]
			for k in range(len(t)):
				kernal[k][i][j] = t[k]

	ans = [[[0 for k in range(14)] for i in range(14)] for j in range(32)]
	
	ans = np.array(ans, dtype = float)

	i = 0
	j = 0
	pos_x = 0
	pos_y = 0
	while i <= padded_image.shape[0] - kernal_x :
		
		pos_y = 0
		j = 0
		while j <= padded_image.shape[1] - kernal_y :
			
			# 4 * 4 image for multiplication
			xx = [im[j:j+kernal_y] for im in padded_image[i:i+kernal_x]]
			req_image = np.array(xx)
			for ker in range(32):
				final = np.multiply(req_image,kernal[ker])
				finalvalue = np.sum(final)
				ans[ker][pos_x][pos_y] = finalvalue + bias[ker]
				# for debugging check 16 values
				if ker == 5 and pos_x == 2 and pos_y == 5:
					print(xx)
					print(i)
					print(j)
					print(kernal[5])
					print(bias[5])
					final = np.multiply(req_image,kernal[5])
					finalvalue = np.sum(final)
					print(final)
					print(finalvalue)
					anss = finalvalue + bias[5]
					print("check the va")
					print(anss)
			j += stride
			pos_y += 1
		pos_x += 1
		i+=stride
	print(ans.shape)

	with open("myresult.txt", "w") as f:
		for ker in ans:
			for row in ker:
				for ele in row:
					f.write(str(ele) + "\n")
	with open("mynewresult_new.txt", "w") as f:
		k_len = len(ans)
		i_len = len(ans[0])
		j_len = len(ans[0][0])
		#as we have to check 1043th value find the 
		#corresponding kernal, i ,j at 1043th value
		countt = 0
		for i in range(0, i_len):
			for j in range(0, j_len):
				for k in range(0, k_len):
					countt += 1
					if countt == 1062:
						print("get the values")
						print(i)
						print(j)
						print(k)
					if(ans[k][i][j] < 0):
						f.write(str(0.0) +"\n")
					else:
						f.write(str(ans[k][i][j]) + "\n")

	

def get_data(dataset):
	csvfile = open('{}_test.csv'.format(dataset), 'r')
	tests = csv.reader(csvfile, delimiter=',')
	return tests

if __name__ == "__main__" :
	argumentList = sys.argv
	netname = argumentList[1]
	filename, file_extension = os.path.splitext(netname)
	#model, _, means, stds = read_tensorflow_net(netname, 784, True)
	#model, d, inp, means, stds = read_tensorflow_net(netname, 784, True)#N
	model, d, inp, means, stds = read_tensorflow_net(netname, 28, True)#N
	#print(model.summary())
	'''-------'''
	#predict = tf.argmax(model, 1)
	dataset=argumentList[2]
	tests = get_data(dataset)
	for i, test in enumerate(tests):
		if i == 7:
			print(d.shape)
			print("correct label:" +test[0])
			lbl = test[0]

			#newtest = np.array(np.float64(test[1:len(test)]))
			newtestt= np.array(np.float64(test[1:len(test)])/np.float64(255))

			#pass @@2D image to tensor

			newtest = np.reshape(newtestt, (-1, 28))

			#Printing  @2D image used 

			imagee = original_image.get_image()
			Y = np.array(np.float64(imagee))
			print(Y)

			rbsaise = np.array(np.float64(test[1:len(test)]))
			bsaise = np.reshape(rbsaise, (-1, 28))
			
			print(bsaise)
			
			print(Y == bsaise)

			#newtest[0] = 1.0
			#newtest[1] = 1.0
			#--------other methods for evaluating tensor#

			#pred = model.eval({inp:newtest})
			#inp = tf.placeholder(tf.float64, [None, 784])
			
			#--------evalating tensorflow----------------#
			'''for i in range(0,28):
				newtest[0][i]=1.0
				newtest[1][i]=1.0'''
			for j in range(0,24):
				for k in range(0,28):
					newtest[j][k]=1.0
			with tf.Session() as sess:
				pred = sess.run(model, feed_dict={inp: newtest})
			get_next_layer(d)
			
			#pred = model.eval({inp:newtest})
			myset = set()
			bias = original_image.get_bias()
			count = 0
			with open("predict_new.txt", "w") as f:
				for fir in pred:
					for sec in fir:
						for thi in sec:
							for fo in thi:
								#count += 1
								f.write(str(fo) + "\n")
								#myset.add(fo)
								#if fo not in bias and fo != 0:
									#print(count)

			#print(myset)'''
			

			'''with open("predict.txt", "wb") as fp:
				pickle.dump(pred, fp)
			#print(pred)
			with open("predict.txt", "rb") as fp:
				b = pickle.load(fp)'''
			#print(b)
			#print(pred)
			#print(len(pred))
			#print(len(pred[0]))
			#print(len(pred[0][0]))
			#print(len(pred[0][0][0]))
			#newpred = pred.flatten()
			#print(newpred)
			#maximum = np.argmax(newpred)
			#index = np.where(newpred == maximum)
			#print(maximum)
			'''modl = z3_format_converter.solve_cons(lbl)
			image= np.float64(test[1:len(test)])/np.float64(255)
			epsilon = 0.1
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
			with open("testn.txt", "wb") as fp:
				pickle.dump(newimage, fp)
			
			#break'''
	read_n()