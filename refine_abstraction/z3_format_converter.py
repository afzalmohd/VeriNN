from z3 import *
from read_net_file import *
import sys
import csv
import pickle
from parsing import *
def solve_cons(s, eta_set, output_cons, lbl, epsilon, img):
        f = open(output_cons, "r+")
        mode = 1
        nodes = []
        change_to_sat_format(s,eta_set, f, mode, 0, epsilon, 0, 0, img,nodes)
        # print(nodes)   

        # for eta_dash we are doing this
        A=[]
        for i in range(0, 10):
                if i!= int(lbl):
                        A.append(nodes[i] > nodes[int(lbl)])
        s.add(Or(A))
        if s.check() == sat:
                m = s.model()
        else:
                return -1
        # for c in A:
        #         print(c)
        #         print(m.eval(c))
        return m

