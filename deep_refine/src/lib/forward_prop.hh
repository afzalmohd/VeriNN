 #ifndef _FORWARD_PROP_H_
 #define _FORWARD_PROP_H_
 #include "network.hh"

void init_z3_expr_neuron_forward(z3::context &c, Neuron_t* nt);
void init_z3_expr_neuron_first_forward(z3::context &c, Neuron_t* nt);
void init_z3_expr_layer_forward(z3::context &c, Layer_t* layer);
void init_z3_expr_layer_first_forward(z3::context& c, Layer_t* layer);
void init_z3_expr_forward(z3::context &c, Network_t* net);

#endif