#ifndef __PULLBACK__HH__
#define __PULLBACK__HH__

#include "gurobi_c++.h"
#include "../../deeppoly/deeppoly_driver.hh"

bool pull_back_full(Network_t* net);
void pull_back_relu(Layer_t* layer);
bool pull_back_FC(Layer_t* layer);
void create_gurobi_variable(GRBModel& model, std::vector<GRBVar>& var_vector, Layer_t* layer);
void create_layer_constrains_pullback(GRBModel& model, std::vector<GRBVar>& var_vector, Layer_t* layer);
GRBModel create_grb_env_and_model();


#endif