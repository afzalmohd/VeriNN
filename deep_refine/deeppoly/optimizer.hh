#ifndef __OPTIMIZER_HH__
#define __OPTIMIZER_HH__
//#include "/home/u1411251/Documents/Phd/tools/eran/gurobi912/linux64/include/gurobi_c++.h"
#include "gurobi_c++.h"
#include "network.hh"
void compute_bounds_using_gurobi(Network_t* net, Layer_t* layer, Neuron_t* nt, Expr_t* expr, bool is_minimize);


#endif