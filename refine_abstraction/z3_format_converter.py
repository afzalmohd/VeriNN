from z3 import *
from read_net_file import *
import sys
import csv
import pickle

def solve_cons(cons_file_name, lbl):
        f = open(cons_file_name, "r+")
        #set_option(html_mode=False)
        s = Solver()
        nodes = []
        lbound = []
        ubound = []
        ind  = 0
        for line in f:
                if line.find(':=') != -1:
                        y = line.split(':=')
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
                                        newconn = round(float(newconn),4)
                                        expr = expr + RealVal(newconn)
                                else:
                                        coef = newconn.split('.(')[0]
                                        coef = round(float(coef),4)
                                        varr = newconn.split('.(')[1]
                                        var = varr.replace(')', '')
                                        var = Real(var)
                                        s.add(var <= 1 , -1<=var)
                                        expr = expr + (RealVal(coef) * var)
                                i = i + 1
                        s.add(node == expr)
                                
                        # while True:
                        #         try:
                        #                 newconn = newcon[i].strip()
                        #                 if newconn.find("eps") == -1:
                        #                         expr += newconn
                        #                         i = i + 1
                        #                         expr += '+'
                        #                 else:
                        #                         coef = newconn.split('.(')[0]
                        #                         varr = newconn.split('.(')[1]
                        #                         var = varr.replace(')', '')
                        #                         var = Real(var)
                        #                         s.add(var <= 1 , -1<=var)
                        #                         expr += coef * var
                        #                         expr += '+'
                        #                         i = i + 1
                        #         except:
                        #                 s.add(node == expr)
                        #                 break
                else:
                        lb = RealVal( line.split(',')[0] )
                        ubb =  line.split(',')[1]
                        ub = RealVal( ubb.strip('\n') )
                        lbound.append(lb)
                        ubound.append(ub)
                        s.add(nodes[ind] <= ub, nodes[ind] >= lb)
                        ind = ind + 1
                        
        #for eta_dash we are doing this#
        A=[]
        for i in range(0, 10):
                if i!= int(lbl):
                        A.append(nodes[i] >= nodes[int(lbl)])
        s.add(Or(A))
        print(s.check())
        #set_option(max_args=10000000, max_lines=1000000, max_depth=10000000, max_visited=1000000)
        m = s.model()

        # for c in A:
        #         print(c)
        #         print(m.eval(c))
        # for a in m:
        #         print(a)
        #         print(m[a])
        # exit()
        #newm = sorted ([(d, m[d]) for d in m], key = lambda x: (len(str(x[0])), str(x[0])))
        return m
