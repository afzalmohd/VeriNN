from z3 import *
from read_net_file import *
import sys
import csv
import pickle
from z3_format_converter import *
from parsing import *
#input layer#
def initial(modl,s, l):
	for i in range(0, l):
		t = 'eps' + str(i)
		t_new = t + 'dd'
		t = Real(t)
		t_new = Real(t_new)
		#print(modl[t])
		s.add(t_new == modl[t])
def solve_cons_inner(s,siz1, siz2, ls_obj, internal_cons, eta_set):
	
	f = open(internal_cons, "r+")
	nodes = []
	change_to_sat_format(s,eta_set, f, 2, 0,0, siz1, siz2, ls_obj, 0, nodes)
	print(sorted(eta_set))
	print(len(eta_set))
def solve_cons_out(s,ls_obj,m,l,l_max, lbl, output_cons, eta_set):
	f = open(output_cons, "r+")
	nodes = []
	change_to_sat_format(s,eta_set, f, 3, 2, 0, 0,0, ls_obj, 0, nodes)
	
	# A=[]
	# for i in range(0, 10):
	# 	if i!= int(lbl):
	# 		A.append(nodes[i] >= nodes[int(lbl)])
	# s.add(Or(A))
	
	'''for c in A:
    	print(c)
        print(m.eval(c))'''
	if s.check() == sat:
		m_new = s.model()
		return 1
	else:
		return -11
	'''for i in range(l, l_max):
		t = 'eps' + str(i)
		t_new = t + 'dd'
		t = Real(t)
		t_new = Real(t_new)
		s.add_soft(t == m_new[t_new])
	if s.check() == sat:
		return s.model
	else:
		return -1'''
#if __name__ == "__main__" :
	#solve_cons()