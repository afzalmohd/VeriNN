#ifndef _DEEPPOLY_ANALYSIS_H_
#define _DEEPPOLY_ANALYSIS_H_
#include "network.hh"
#include "gurobi_c++.h"
#include<unordered_set>
bool forward_analysis(Network_t* net);
bool milp_based_deeppoly(Network_t* net, Layer_t* marked_layer);
bool forward_layer_milp(Network_t* net, Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter);
void milp_layer_FC_parallel(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& pred_layer_vars, std::vector<GRBVar>& var_vector, size_t var_counter);
bool milp_layer_FC(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& pred_layer_vars, std::vector<GRBVar>& var_vector, size_t var_counter, size_t start_index, size_t end_index);
bool set_neurons_bounds(Layer_t* layer, Neuron_t* nt, GRBModel& model, bool is_lower);
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
bool is_image_verified_deeppoly(Network_t* net);
bool is_greater(Network_t* net, size_t index1, size_t index2, bool is_stricly_greater);
bool is_image_verified_milp(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector);
bool is_sat_property_main(Network_t* net);
bool is_sat_with_milp(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector, Vnnlib_post_cond_t* conj_cond, bool is_first);
void remove_constr_grb_model(GRBModel& model, std::vector<GRBConstr>& constr_vec);
void set_basic_cond_constr(Network_t* net, GRBModel& model, std::vector<GRBConstr>& constr_vec, std::vector<GRBVar>& var_vector, std::unordered_set<size_t>& indexes_in_prp, Basic_post_cond_t* basic_cond);
void set_rel_cond_constr(Network_t* net, GRBModel& model, std::vector<GRBConstr>& constr_vec, std::vector<GRBVar>& var_vector, std::unordered_set<size_t>& indexes_in_prp, Basic_post_cond_t* basic_cond);
bool is_sat_prop_main_pure_milp(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vec);
bool is_sat_property_conj(Network_t* net, Vnnlib_post_cond_t* conj_cond);
bool is_verified_neg_rel_property(Network_t* net, Basic_post_cond_t* basic_cond);
bool is_verified_neg_basic_property(Network_t* net, Basic_post_cond_t* basic_cond);
bool is_verified_single_nt_bound(Network_t* net, size_t nt_index, double bound, bool is_upper, bool is_strict_cond);
std::string get_neg_op(std::string op);
bool is_prop_sat_vnnlib(Network_t* net);
bool is_prop_sat_vnnlib_conj(Network_t* net, Vnnlib_post_cond_t* prop);
bool is_basic_prop_sat(Network_t* net, Basic_post_cond_t* basic_cond);
bool is_rel_prop_sat(Network_t* net, Basic_post_cond_t* basic_cond);
void print_xt_array(xt::xarray<double> x_arr, size_t size);
std::vector<size_t> get_max_elems_indexes_vec(Network_t* net, std::vector<double>& vec);
bool is_val_exist_in_vec_double(double val, std::vector<double>& vec);
bool is_no_ce_with_conf(Network_t* net);


#endif