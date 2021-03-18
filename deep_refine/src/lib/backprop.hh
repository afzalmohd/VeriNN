 #ifndef _BACKPROP_H_
 #define _BACKPROP_H_
 #include "network.hh"

 void back_substitute_neuron(z3::context &c, Neuron_t* nt);
 void back_substitute_layer(z3::context& c, Layer_t* layer);
 void back_substitute(z3::context& c, Network_t* net);

#endif
