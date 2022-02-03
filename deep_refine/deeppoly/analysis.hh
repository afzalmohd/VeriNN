#ifndef _DEEPPOLY_ANALYSIS_H_
#define _DEEPPOLY_ANALYSIS_H_
#include "network.hh"
#include "gurobi_c++.h"
bool forward_analysis(Network_t* net);
bool milp_based_deeppoly(Network_t* net, Layer_t* marked_layer);
void forward_layer_milp(Network_t* net, Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter);
void milp_layer_FC_parallel(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& pred_layer_vars, std::vector<GRBVar>& var_vector, size_t var_counter);
void milp_layer_FC(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& pred_layer_vars, std::vector<GRBVar>& var_vector, size_t var_counter, size_t start_index, size_t end_index);
void set_neurons_bounds(Layer_t* layer, Neuron_t* nt, GRBModel& model, bool is_lower);
void forward_layer_FC_parallel(Network_t* net, Layer_t* curr_layer);
void forward_layer_FC(Network_t* net, Layer_t* curr_layer, size_t start_index, size_t end_index);
void forward_layer_ReLU_parallel(Network_t* net, Layer_t* curr_layer);
void forward_layer_ReLU(Network_t* net, Layer_t* curr_layer, size_t start_index, size_t end_index);
void update_neuron_relu(Network_t* net, Layer_t* pred_layer, Neuron_t* nt);
void update_relu_expr(Neuron_t* curr_nt, Neuron_t* pred_nt, bool is_default_heuristic, bool is_lower);
void update_neuron_FC(Network_t* net, Layer_t* layer, Neuron_t* nt);
void create_marked_layer_splitting_constraints(Layer_t* layer);
void create_neuron_expr_FC(Neuron_t* net, Layer_t* layer);
void update_neuron_lexpr_bound_back_substitution(Network_t* net, Layer_t* pred_layer, Neuron_t* nt);
void update_neuron_uexpr_bound_back_substitution(Network_t* net, Layer_t* pred_layer, Neuron_t* nt);
void update_neuron_bound_back_substitution(Network_t* net, Layer_t* pred_layer, Neuron_t* nt);
void update_neuron_lexpr_b(Network_t* net, Layer_t* pred_layer, Neuron_t* nt);
Expr_t* update_expr_affine_backsubstitution(Network_t* net, Layer_t* pred_layer,Expr_t* curr_expr, Neuron_t* curr_nt, bool is_lower);
Expr_t* update_expr_relu_backsubstitution(Network_t* net, Layer_t* pred_layer, Expr_t* curr_expr, Neuron_t* nt, bool is_lower);
void create_input_layer_expr(Network_t* net);
double compute_lb_from_expr(Layer_t* pred_layer, Expr_t* expr);
double compute_ub_from_expr(Layer_t* pred_layer, Expr_t* expr);
bool is_image_verified(Network_t* net);
bool is_greater(Network_t* net, size_t index1, size_t index2);
bool is_image_verified_milp(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector);
#endif