#ifndef __CONCURRENT_RUN_HH__
#define __CONCURRENT_RUN_HH__
#include "../../deeppoly/network.hh"
#include "gurobi_c++.h"
#include "../../deeppoly/deeppoly_configuration.hh"
#include <bits/stdc++.h>
#include <pthread.h>

bool concurrent_exec(Network_t *net);
std::vector<std::vector<int>> combsgen(int n);
void helper(int n, std::vector<int> &curr, std::vector<std::vector<int>> &result);
void *multi_thread(void *p);
bool model_gen(std::vector<int> &activations, int index,std::vector<Neuron_t*> new_list_mn);
bool add_constraint(Network_t *net, GRBModel &model, std::vector<GRBVar> &var_vector, std::vector<int> &activations, int index,std::vector<Neuron_t*> new_list_mn);
int relu_constr_mine(Layer_t *layer, GRBModel &model, std::vector<GRBVar> &var_vector, size_t var_counter, std::vector<int> &activations,int marked_index,std::vector<Neuron_t*> new_list_mn);
bool rec_con(Network_t *net, std::vector<int> prev_comb, int new_mn);

extern pthread_t thread_id[NUM_THREADS];
// extern concurr_status status_network;
extern bool is_refine;
extern std::vector<int> refine_comb;
extern volatile sig_atomic_t terminate_flag;
#endif