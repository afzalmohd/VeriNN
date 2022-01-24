#ifndef __OPTIMIZER_HH__
#define __OPTIMIZER_HH__
//#include "/home/u1411251/Documents/Phd/tools/eran/gurobi912/linux64/include/gurobi_c++.h"
#include "network.hh"
#include "gurobi_c++.h"

void compute_bounds_using_gurobi(Network_t* net, Layer_t* layer, Neuron_t* nt, Expr_t* expr, bool is_minimize);
GRBModel create_env_model_constr(Network_t* net, std::vector<GRBVar>& var_vector);
bool verify_by_milp(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector, size_t counter_class_index);
void creating_variables(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector);
void create_milp_constr_FC(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter);
void create_milp_constr_relu(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter);
void copy_vector_by_index(std::vector<GRBVar>& var_vector, std::vector<GRBVar>& new_vec, size_t start_index, size_t end_index);
size_t get_gurobi_var_index(Layer_t* layer, size_t index);
GRBModel create_env_and_model();
void update_sat_vals(Network_t* net, std::vector<GRBVar>& var_vec);


#endif