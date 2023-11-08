#ifndef __VERICOMP__
#define __VERICOMP__
#include "../../deeppoly/network.hh"
#include "../../deeppoly/deeppoly_configuration.hh"

drefine_status is_verified_by_vericomp(Network_t* net);
void parse_file_and_update_bounds(Network_t* net, std::string &file_path);


#endif