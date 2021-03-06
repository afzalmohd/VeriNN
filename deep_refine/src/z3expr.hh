#include "network.hh"


void set_predecessor_layer_activation(z3::context& c, Layer_t* layer, Layer_t* prev_layer);
void set_predecessor_layer_matmul(z3::context& c, Layer_t* layer, Layer_t* prev_layer);
void set_predecessor_and_z3_var(z3::context &c, Network_t* net);
z3::expr get_expr_from_double(z3::context &c, double item);
void init_z3_expr_neuron(z3::context &c, Neuron_t* nt);
void init_z3_expr_layer(z3::context &c, Layer_t* layer);
void init_z3_expr(z3::context &c, Network_t* net);