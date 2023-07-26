#ifndef __HELPER_HH__
#define __HELPER_HH__
#include "network.hh"

unsigned int get_num_thread();
void copy_layer_constraints(Layer_t* layer, Neuron_t* nt);
void update_pred_layer_link(Network_t* net, Layer_t* pred_layer);
void add_expr(Network_t* net, Expr_t* expr1, Expr_t* expr2);
Expr_t* multiply_expr_with_coeff(Network_t* net, Expr_t* expr, double coeff_inf, double coeff_sup);
Layer_t* get_pred_layer(Network_t* net, Layer_t* curr_layer);
Expr_t* get_mul_expr(Neuron_t* pred_nt, double inf_coff, double supp_coff, bool is_lower);
void create_marked_layer_splitting_constraints(Layer_t* layer);
// void create_constr_vec_by_size(std::vector<Constr_t*>& constr_vec, std::vector<Constr_t*>& old_vec, size_t constr_size);
// void create_constr_vec_with_init_expr(std::vector<Constr_t*>& constr_vec, std::vector<Constr_t*>& old_vec, size_t constr_size);
// void update_independent_constr_relu(Network_t* net, std::vector<Constr_t*>& new_constr_vec, std::vector<Constr_t*>& old_constr_vec,Neuron_t* pred_nt);
// void update_dependent_constr_relu(Network_t* net, std::vector<Constr_t*>& new_constr_vec, std::vector<Constr_t*>& old_constr_vec, Expr_t* mul_expr, Neuron_t* pred_nt);
// void update_independent_constr_FC(Network_t* net, std::vector<Constr_t*>& new_constr_vec, std::vector<Constr_t*>& old_constr_vec,Neuron_t* pred_nt);
// void update_dependent_constr_FC(Network_t* net, std::vector<Constr_t*>& new_constr_vec, std::vector<Constr_t*>& old_constr_vec, Expr_t* mul_expr, Neuron_t* pred_nt);
// void update_constr_vec_cst(std::vector<Constr_t*> new_constr_vec, std::vector<Constr_t*>& old_constr_vec);
// void free_constr_vector_memory(std::vector<Constr_t*>& constr_vec);
void copy_vector_with_negative_vals(std::vector<double> &vec1, std::vector<double> &vec2);
std::vector<double> get_neuron_incomming_weigts(Neuron_t* nt, Layer_t* layer);
double get_neuron_bias(Neuron_t* nt, Layer_t* layer);
void create_input_property_vnnlib(Network_t* net, Basic_pre_cond_t* pre_cond);
void update_last_layer(Network_t* net);
//Constr_t* declare_constr_t();
#endif