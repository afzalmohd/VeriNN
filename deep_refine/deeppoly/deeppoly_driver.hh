#ifndef __DEEPPOLY_DRIVER__
#define __DEEPPOLY_DRIVER__

#include "network.hh"

int deeppoly_set_params(int argc, char* argv[]);
Network_t* deeppoly_initialize_network();
void deeppoly_parse_input_image_string(Network_t* net, std::string & image_str);
void deeppoly_reset_network(Network_t* net);
size_t execute_network(Network_t* net);
int run_deeppoly(Network_t* net);

#endif