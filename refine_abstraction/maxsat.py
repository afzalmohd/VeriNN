from z3 import *
from read_net_file import *
import sys
import csv
import pickle
from z3_format_converter import *
#input layer#
def initial(modl,s, l):
	for i in range(0, l):
		t = 'eps' + str(i)
		t_new = t + 'dd'
		t = Real(t)
		t_new = Real(t_new)
		#print(modl[t])
		s.add(t_new == modl[t])
def solve_cons_inner(s,siz1, siz2, ls_obj, internal_cons):
	f = open(internal_cons, "r+")
	#set_option(html_mode=False)
	nodes = []
	lbound = []
	ubound = []
	ind  = 0
	ls_i = 0
	layer = 0
	for x in f:
		if x.find(':=') != -1:
			y = x.split(':=')
			node = (y[0].strip())
			
			node = Real(node)
			nodes.append(node)
			con = (y[1].strip())
			newcon = con.split('+')
			i = 0

			expr = 0
			while i < len(newcon):
				newconn = newcon[i].strip()
				if newconn.find("eps") == -1:
					expr = expr + RealVal(newconn)
				else:
					coef = newconn.split('.(')[0]
					varr = newconn.split('.(')[1]
					var = varr.replace(')', '')
					var = var + 'dd'
					
					var = Real(var)
					
					s.add(var <= 1 , -1<=var)
					expr = expr + (RealVal(coef) * var)
				i = i + 1
			s.add(node == expr)
			
			'''
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
						var = var + 'dd'
						var = Real(var)
						s.add(var <= 1 , -1<=var)
						expr += coef * var
						expr += '+'
						i = i + 1
				except:
					s.add(node == expr)
					break
			'''
		else:
			lb = RealVal(x.split(',')[0] )
			ubb = x.split(',')[1]
			ub = RealVal(ubb.strip('\n') )
			lbound.append(lb)
			ubound.append(ub)
			
			s.add(nodes[ind] == ls_obj[layer][ls_i])
			ls_i = ls_i + 1
			if ls_i == siz1:
				ls_i=0
				layer+=1
			
			ind = ind + 1
	
def solve_cons_out(s,ls_obj,m,l,l_max, lbl, output_cons):
	f = open(output_cons, "r+")
	set_option(html_mode=False)
	nodes = []
	lbound = []
	ubound = []
	ind  = 0
	ls_i = 0
	layer = 2
	for x in f:
		if x.find(':=') != -1:
			y = x.split(':=')
			node = (y[0].strip())
			node = Real(node)
			nodes.append(node)
			con = (y[1].strip())
			newcon = con.split('+')
			i = 0

			expr = 0
			while i < len(newcon):
				newconn = newcon[i].strip()
				if newconn.find("eps") == -1:
					expr = expr + RealVal(newconn)
				else:
					coef = newconn.split('.(')[0]
					varr = newconn.split('.(')[1]
					var = varr.replace(')', '')
					var = var + 'dd'
					
					var = Real(var)
					
					s.add(var <= 1 , -1<=var)
					expr = expr + (RealVal(coef) * var)
				i = i + 1
			s.add(node == expr)

			'''
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
						var = var + 'dd'
						var = Real(var)
						s.add(var <= 1 , -1<=var)
						expr += coef * var
						expr += '+'
						i = i + 1
				except:
					s.add(node == expr)
					break
			'''
		else:
			lb = RealVal(x.split(',')[0] )
			ubb = x.split(',')[1]
			ub = RealVal(ubb.strip('\n') )
			lbound.append(lb)
			ubound.append(ub)
			s.add(nodes[ind] == ls_obj[layer][ls_i])
			ind = ind + 1
			ls_i = ls_i + 1
	
	A=[]
	for i in range(0, 10):
		if i!= int(lbl):
			A.append(nodes[i] >= nodes[int(lbl)])
	s.add(Or(A))
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