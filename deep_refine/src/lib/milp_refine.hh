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
void create_milp_mark_milp_refine_constr_ab(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector);
void create_milp_constr_FC_without_marked_ab(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter);
bool is_prp_verified_ab(Network_t* net);
bool is_sat_val_ce(Network_t* net);
void get_marked_neurons(GRBModel& model,  Network_t* net, std::vector<GRBVar>& var_vector);
bool is_layer_marked_after_optimization(Layer_t* start_layer, std::vector<GRBVar>& var_vector, size_t var_counter);
void update_vars_bounds(Layer_t* layer, std::vector<GRBVar>& var_vector, size_t var_counter);
bool run_refinement_cegar(Network_t* net);
void update_vars_bounds_by_prev_satval(Layer_t* layer, std::vector<GRBVar>& var_vector, size_t var_counter);
void remove_maxsat_constr(GRBModel& model, Layer_t* layer);
void create_exact_relu_constr_milp_refine(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter);
bool is_layer_marked_after_optimization_without_maxsat(Layer_t* start_layer);
double compute_softmax_conf(Network_t* net, size_t label);
double compute_conf(Network_t* net, size_t label);
#endif