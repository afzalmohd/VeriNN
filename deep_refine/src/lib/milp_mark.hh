#ifndef __MILP_MARK__
#define __MILP_MARK__
#include "../../deeppoly/network.hh"
#include "gurobi_c++.h"

bool run_milp_mark_with_milp_refine(Network_t* net);
bool mark_neurons_with_light_analysis(Network_t* net);
bool is_layer_marked(Network_t* net, Layer_t* start_layer);
void create_optimization_constraints_layer(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter);
void creating_vars_with_constant_vars(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector, size_t start_layer_index);
void create_constant_vars_satval_layer(Network_t* net, Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector);
void create_relu_constr(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter);
void create_satvals_to_image(Layer_t* layer);
void get_images_from_satval(xt::xarray<double>& res, Layer_t* layer);
std::string get_consr_name_binary(size_t layer_idx, size_t nt_idx);
Neuron_t* get_key_of_max_val(std::map<Neuron_t*, double> & m);
GRBModel create_grb_env_and_model();
void get_marked_neurons_reverse(Network_t* net);


#endif