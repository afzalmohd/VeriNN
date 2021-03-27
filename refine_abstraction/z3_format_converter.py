from z3 import *
from read_net_file import *
import sys
import csv
import pickle
from parsing import *
def solve_cons(s, eta_set, output_cons, lbl, epsilon, img):
        f = open(output_cons, "r+")
        nodes = []
        change_to_sat_format(s,eta_set, f, 1, 0, epsilon, 0, 0, 0, img,nodes)
        print(nodes)   
        #for eta_dash we are doing this#
        A=[]
        for i in range(0, 10):
                if i!= int(lbl):
                        A.append(nodes[i] >= nodes[int(lbl)])
        s.add(Or(A))
        print(s.check())
        #set_option(max_args=10000000, max_lines=1000000, max_depth=10000000, max_visited=1000000)
        m = s.model()

        for c in A:
                print(c)
                print(m.eval(c))
        # print(m)
        #exit()
        #newm = sorted ([(d, m[d]) for d in m], key = lambda x: (len(str(x[0])), str(x[0])))
        return m

