#ifndef __MILP_MARK__
#define __MILP_MARK__
#include "../../deeppoly/network.hh"
#include "gurobi_c++.h"

bool run_milp_mark_with_milp_refine(Network_t* net);
bool is_layer_marked(Network_t* net, Layer_t* start_layer);
void create_optimization_constraints_layer(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter);
void creating_vars_with_constant_vars(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector, size_t start_layer_index);
void create_constant_vars_satval_layer(Network_t* net, Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector);
void create_relu_constr(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter);
bool is_sat_val_ce(Network_t* net);
void create_satvals_to_image(Layer_t* layer);
// void create_negate_property(GRBModel& model, std::vector<GRBVar>& var_vector, Network_t* net, Layer_t* curr_layer);

#endif