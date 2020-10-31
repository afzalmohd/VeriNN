from z3 import *
from read_net_file import *
import sys
import csv
import pickle
def solve_cons(lbl):
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
	A=[]
	for i in range(0, 10):
		if i!= lbl:
			A.append(nodes[i] > nodes[int(lbl)])
	s.add(Or(A))
	print(s.check())
	set_option(max_args=10000000, max_lines=1000000, max_depth=10000000, max_visited=1000000)
	m = s.model()
	#newm = sorted ([(d, m[d]) for d in m], key = lambda x: (len(str(x[0])), str(x[0])))
	return m