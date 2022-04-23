#ifndef __MILP_REFINE__
#define __MILP_REFINE_
#include "gurobi_c++.h"
#include "../../deeppoly/network.hh"
bool is_image_verified_by_milp(Network_t* net);
void create_milp_mark_milp_refine_constr(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector);
void creating_vars_milp(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector);
void create_relu_constr_milp_refine(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter);
void create_vars_layer(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector);
void create_milp_constr_FC_without_marked(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter);
bool is_prp_verified_by_milp(Network_t* net);
#endif