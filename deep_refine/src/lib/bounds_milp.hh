#ifndef __BOUNDS__MILP__
#define __BOUNDS_MILP__
#include "../../deeppoly/network.hh"
#include "gurobi_c++.h"

void forward_analysis_bounds_milp_seq(Network_t* net);
void update_bounds(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter);
bool is_verified_by_bound_tighten_milp(Network_t* net);


#endif