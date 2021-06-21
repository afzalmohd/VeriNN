#ifndef _Z3EXPR_H_
#define _Z3EXPR_H_
#include "network.hh"

void set_z3_parameters(z3::context& c);
void set_predecessor_layer_activation(z3::context& c, Layer_t* layer, Layer_t* prev_layer);
void set_predecessor_layer_matmul(z3::context& c, Layer_t* layer, Layer_t* prev_layer);
void set_predecessor_and_z3_var(z3::context &c, Network_t* net);
void check_sat_output_layer(z3::context& c, Network_t* net);
void model_to_image(z3::model &modl, Network_t* net);
void init_z3_expr(z3::context& c, Network_t* net);
void init_z3_expr_layer(z3::context& c, Layer_t* layer);
void init_z3_expr_neuron(z3::context &c, Neuron_t* nt);
void merged_constraints(z3::context& c, Network_t* net);
void sat_var_value(z3::context& c, z3::model &modl, Layer_t* layer);
void affine_expr_init_neuron(z3::context& c, Layer_t* layer, Neuron_t* nt);
void affine_expr_init_layer(z3::context& c, Layer_t* layer);
void affine_expr_init(z3::context& c, Network_t* net);
void back_prop_relu(z3::context& c, Network_t* net, Layer_t* layer);
#endif