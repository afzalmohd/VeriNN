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
sys.setrecursionlimit(1000000000)
def read_n():
	pass
def get_data(dataset):
	csvfile = open('{}_test.csv'.format(dataset), 'r')
	tests = csv.reader(csvfile, delimiter=',')
	return tests
if __name__ == "__main__" :
	argumentList = sys.argv
	netname = argumentList[1]
	filename, file_extension = os.path.splitext(netname)
	#model, _, means, stds = read_tensorflow_net(netname, 784, True)
	model, inp, means, stds = read_tensorflow_net(netname, 784, True)
	'''-------'''
	#predict = tf.argmax(model, 1)
	dataset=argumentList[2]
	tests = get_data(dataset)
	for i, test in enumerate(tests):
		if i == 7:
			print("correct label:" +test[0])
			lbl = test[0]
			#newtest = np.array(np.float64(test[1:len(test)]))
			newtest= np.array(np.float64(test[1:len(test)])/np.float64(255))
			#pred = model.eval({inp:newtest})
			#inp = tf.placeholder(tf.float64, [None, 784])
			with tf.Session() as sess:
				pred = sess.run(model, feed_dict={inp: newtest})
			#pred = model.eval({inp:newtest})
			print(pred)
			print(len(pred[0]))
			#newpred = pred.flatten()
			#print(newpred)
			#maximum = np.argmax(newpred)
			#print("correct:" +test[0] + "class :" + str(maximum))
			#index = np.where(newpred == maximum)
			#print(index)
			modl = z3_format_converter.solve_cons(lbl)
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
			
			#break
	read_n()