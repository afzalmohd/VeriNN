#ifndef __DECISION_MAKING__
#define  __DECISION_MAKING__
#include "../../deeppoly/deeppoly_driver.hh"
#include "gurobi_c++.h"

bool marked_neurons_vector(Network_t* net, std::vector<std::vector<Neuron_t*>>& marked_nt);
bool is_duplicate_neuron_marked(Network_t* net, Neuron_t* nt);
bool set_marked_path(Network_t* net, std::vector<std::vector<Neuron_t*>>& marked_vec, bool is_first);
bool set_pred_path_if_not_valid(Network_t* net, std::vector<std::vector<Neuron_t*>>& marked_vec);
bool is_valid_path(Network_t* net, std::vector<Neuron_t*>& nt_vec);
void reset_mark__nt(Network_t* net, std::vector<Neuron_t*>& nt_vec);
bool set_to_predecessor(std::vector<Neuron_t*>& nt_vector, bool is_first);
bool is_valid_path_with_iss(Layer_t* layer);
bool is_valid_path_with_milp(Layer_t* layer);
void create_layer_constrains_for_valid_path(GRBModel& model, std::vector<GRBVar>& var_vector, Layer_t* layer);
void create_gurobi_variable_with_unmarked_bounds(GRBModel& model, std::vector<GRBVar>& var_vector, Layer_t* layer);
#endif
