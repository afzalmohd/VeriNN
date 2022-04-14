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
VerinnLib_t* parse_vnnlib_file(std::string& prop_file);
void init_bound_vecs(size_t max_inp_index, size_t max_out_index, std::vector<double>& inp_lb, std::vector<double>& inp_ub, std::vector<double>& out_lb, std::vector<double>& out_ub);
void get_vars(std::cmatch& m_var, size_t& max_index_in_vars, size_t& max_index_out_vars, size_t& num_in_vars, size_t& num_out_vars);
void parse_constraints_vnnlib(std::cmatch& m_var, std::vector<double>& in_lb, std::vector<double>& in_ub, std::vector<double>& out_lb, std::vector<double>& out_ub);

#endif