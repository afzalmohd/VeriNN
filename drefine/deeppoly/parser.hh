#ifndef _DEEPPOLY_PARSER_H_
#define _DEEPPOLY_PARSER_H_
#include "network.hh"
#include<regex>
#include<vector>

void init_network(Network_t* net, std::string &filepath);
void pred_layer_linking(Network_t* net);
void parse_string_to_xarray(Layer_t* layer, std::string weights, bool is_bias);
Layer_t* create_layer(bool is_activation, std::string activation, std::string layer_type);
void create_neurons_update_layer(Layer_t* layer);
Layer_t* create_input_layer(size_t dim);
void parse_input_image(Network_t* net, std::string &image_path, size_t image_index);
void parse_image_string_to_xarray_one(Network_t* net, std::string &image_str);
void parse_vnnlib_simplified_mnist(Network_t* net, std::string& file_path);
void bounds_parser(Network_t* net, std::string& file_path);
#endif