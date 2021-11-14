#ifndef _DEEPPOLY_ANALYSIS_H_
#define _DEEPPOLY_ANALYSIS_H_
#include "network.hh"
void create_input_layer_expr(Network_t* net);
void forward_layer_FC(Network_t* net, int layer_index);
void create_neuron_expr_FC(Neuron_t* net, Layer_t* layer);
void update_neuron_bound_back_substitution(Network_t* net, int layer_index, Neuron_t* nt);
double compute_lb_from_expr(Layer_t* pred_layer, Expr_t* expr);
double compute_ub_from_expr(Layer_t* pred_layer, Expr_t* expr);

#endif