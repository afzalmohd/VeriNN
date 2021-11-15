#ifndef _DEEPPOLY_ANALYSIS_H_
#define _DEEPPOLY_ANALYSIS_H_
#include "network.hh"
void create_input_layer_expr(Network_t* net);
void forward_layer_FC(Network_t* nt, int layer_index);
void forward_layer_ReLU(Network_t* net, Layer_t* curr_layer);
void update_neuron_relu(Network_t* net, Layer_t* pred_layer, Neuron_t* nt);
void update_relu_expr(Neuron_t* curr_nt, Neuron_t* pred_nt, bool is_default_heuristic, bool is_lower);
void update_neuron_FC(Network_t* net, Layer_t* layer, Neuron_t* nt);
void create_neuron_expr_FC(Neuron_t* net, Layer_t* layer);
void update_neuron_bound_back_substitution(Network_t* net, int layer_index, Neuron_t* nt);
void update_neuron_lexpr_b(Network_t* net, Layer_t* pred_layer, Neuron_t* nt);
Expr_t* update_expr_affine_backsubstitution(Network_t* net, Layer_t* pred_layer,Expr_t* curr_expr, Neuron_t* curr_nt, bool is_lower);
Expr_t* update_expr_relu_backsubstitution(Network_t* net, Layer_t* pred_layer, Expr_t* curr_expr, bool is_lower);
Expr_t* get_mul_expr(Neuron_t* pred_nt, double inf_coff, double supp_coff, bool is_lower);
double compute_lb_from_expr(Layer_t* pred_layer, Expr_t* expr);
double compute_ub_from_expr(Layer_t* pred_layer, Expr_t* expr);
Layer_t* get_pred_layer(Network_t* net, Layer_t* curr_layer);
Expr_t* multiply_expr_with_coeff(Network_t* net, Expr_t* expr, double coeff_inf, double coeff_sup);
void add_expr(Network_t* net, Expr_t* expr1, Expr_t* expr2);

#endif