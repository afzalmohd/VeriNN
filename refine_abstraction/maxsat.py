from z3 import *
from read_net_file import *
import sys
import csv
import pickle
from z3_format_converter import *
from parsing import *
#input layer#
def initial(modl,_solver, image_size):
	for i in range(0, image_size):
		t = 'eps' + str(i)
		t_new = t + 'dd'
		t = Real(t)
		t_new = Real(t_new)
		_solver.add(t_new == modl[t])
def solve_cons_inner(_solver,layerwise_size, ls_val, internal_cons, eta_set):
	
	f = open(internal_cons, "r+")
	nodes = []
	
	'''

	calling a parsing function which read internal_cons
	and add some other cons in _solver to give eta_dd
	values.
	format:
	change_to_sat_format(solver, etas, file_descripter, mode,
	layer_number - 1, no use, layer sizes, values of 
	internal nodes, no use, list of nodes explored)

	'''

	change_to_sat_format(_solver,eta_set, f, 2, 0, 0,layerwise_size, ls_val, 0, nodes)
	
	
def solve_cons_out(s,layerwise_size, eta_set, eta_dd ,lbl, output_cons, internal_cons):
	f = open(internal_cons, "r+")
	
	nodes = []
	nodes_out = []
	'''
        debugging values of each node by
        putting eta_dd values into eqn...
        so we created a new solver
    '''
	debug_solver = Solver()
	'''
	inner layers
	
	'''
	add_maxsat_cons(debug_solver, s, eta_set, eta_dd, f, 2, 0, layerwise_size, nodes)
	f_new = open(output_cons, "r+")
	add_maxsat_cons(debug_solver, s, eta_set, eta_dd, f_new, 3, 2, layerwise_size, nodes_out)

	# change_to_sat_format(s,eta_set, f, 4, 2, 0, 0,0, ls_obj, 0, nodes)
	print(eta_dd)
	for key in eta_dd:
		# print (key[:-2], 'corresponds to', eta_dd[key])
		var = key[:-2]
		var = Real(var)
		debug_solver.add(var == eta_dd[key])
	#---------------debugging----------------------
	
	'''
	debugging for checking
	values at each internal
	layers

	'''
	layer_1 = []
	layer_2 = []
	if debug_solver.check() == sat:
		m = debug_solver.model()
	for x in range(0, 50):
		t = "(" + str(x) + ")"+ "_0"
		u = "(" + str(x) + ")"+ "_1"
		t = Real(t)
		u = Real(u)
		print(t)
		res = m[t]
		res2 = m[u]
		value = float(res.numerator_as_long())/float(res.denominator_as_long())
		value2 = float(res2.numerator_as_long())/float(res2.denominator_as_long())
		layer_1.append(value)
		layer_2.append(value2)
	print(layer_1)
	print(layer_2)
	output_layer = []
	for x in range(50, 60):
		t = "(" + str(x) + ")"
		t = Real(t)
		print(t)
		res = m[t]
		value = float(res.numerator_as_long())/float(res.denominator_as_long())
		output_layer.append(value)
	print(output_layer)

	#------------debugging ends-----------------------
	
	A=[]

	for i in range(0, 10):
		if i!= int(lbl):
			A.append(nodes_out[i] >= nodes_out[int(lbl)])
	print(A)
	s.add(Or(A))
	print(s.check())
	m_new = s.model()
	for c in A:
		print(c)
		print(m_new.eval(c))
	
	
	return m_new