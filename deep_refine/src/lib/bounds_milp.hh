#ifndef __BOUNDS__MILP__
#define __BOUNDS_MILP__
#include "../../deeppoly/network.hh"
#include "gurobi_c++.h"

void forward_analysis_bounds_milp_seq(Network_t* net);
void forward_analysis_bounds_milp_parallel(Network_t* net);
void bounds_tighting_by_milp(Network_t* net);

#endif