from z3 import *
from read_net_file import *
import sys
import csv
import pickle
def eta_double_dash(activations):
	f = open("modified_cons.txt", "r+")
	set_option(html_mode=False)
	s = Solver()
	nodes = []
	lbound = []
	ubound = []
	ind  = 0
	for x in f:
		if x.find(':=') != -1:
			y = x.split(':=')
			node = (y[0].strip())
			node = Real(node)
			nodes.append(node)
			con = (y[1].strip())
			newcon = con.split('+')
			i = 0
			expr = ""
			while True:
				try:
					newconn = newcon[i].strip()
					if newconn.find("eps") == -1:
						expr += newconn
						i = i + 1
						expr += '+'
					else:
						coef = newconn.split('.(')[0]
						varr = newconn.split('.(')[1]
						var = varr.replace(')', '')
						var = var + "_d"
						var = Real(var)
						s.add(var <= 1 , -1<=var)
						expr += coef * var
						expr += '+'
						i = i + 1
				except:
					s.add(node == expr)
					break
		else:
			lb = x.split(',')[0]
			ubb = x.split(',')[1]
			ub = ubb.strip('\n')
			lbound.append(lb)
			ubound.append(ub)
			s.add(nodes[ind] <= ub, nodes[ind] >= lb)
			ind = ind + 1

	#for eta_double_dash we are doing this#
	A = []
	for i in range(0, 10):
		s.add(nodes[i] == activations[i])
	print(s.check())
	m = s.model()
	return m