#ifndef __DREFINE_DRIVER__
#define __DREFINE_DRIVER__
#include<string>
#include<tuple>
#include<queue>
#include "../../deeppoly/deeppoly_configuration.hh"
#include "../../deeppoly/deeppoly_driver.hh"
bool is_actual_and_pred_label_same(Network_t* net, size_t image_index);
// drefine_status run_refine_poly_for_one_task(Network_t* net, std::chrono::_V2::system_clock::time_point start_time);
std::string get_image_str(std::string& image_path, size_t image_index);
int run_milp_refinement_with_pullback(Network_t* net, size_t image_index, std::chrono::_V2::system_clock::time_point start_time);
int run_path_split_with_pullback(Network_t* net, size_t image_index, std::chrono::_V2::system_clock::time_point start_time);
int run_milp_refine_with_milp_mark(Network_t* net, size_t image_index, std::chrono::_V2::system_clock::time_point start_time);
void create_ce_and_run_nn(Network_t* net);
void write_to_file(std::string& file_path, std::string& s);
void print_status_string(Network_t* net, size_t tool_status, std::string tool_name, size_t image_index, size_t loop_counter, std::chrono::_V2::system_clock::time_point start_time);
// void print_failed_string(Network_t* net, std::string tool_name, size_t image_index, size_t loop_counter, std::chrono::_V2::system_clock::time_point start_time);
// void print_verified_string(Network_t* net, std::string tool_name, size_t image_index, size_t loop_counter, std::chrono::_V2::system_clock::time_point start_time);
// void print_unknown_string(Network_t* net, std::string tool_name, size_t image_index, size_t loop_counter, std::chrono::_V2::system_clock::time_point start_time);
bool is_real_ce_mnist_cifar10(Network_t* net);
std::string get_absolute_file_name_from_path(std::string & path);
bool is_valid_dataset();
void set_stds_means(Network_t* net);
void normalize_input_image(Network_t* net); //use for the first time
void normalize_image(Network_t* net); //use in between of the execution of the tool
void denormalize_image(Network_t* net);
int run_drefine_vnnlib(Network_t* net);
std::tuple<int, size_t> run_milp_refine_with_milp_mark_vnnlib(Network_t* net);
size_t num_marked_neurons(Network_t* net);
void remove_non_essential_neurons(Network_t* net);
void create_input_prop(Network_t* net);
void copy_network(Network_t* net1, Network_t* net);
void copy_layer(Layer_t* layer1, Layer_t* layer);
drefine_status run_refine_poly(std::queue<Network_t*>& work_q, std::chrono::_V2::system_clock::time_point start_time);
void create_problem_instances(Network_t* net, std::queue<Network_t*>& work_q);
void create_problem_instances_recursive(Network_t* net, std::queue<Network_t*>& work_q, size_t n, size_t a[], size_t i);
void create_one_problem_instances_input_split(Network_t* net, std::queue<Network_t*>& work_q, size_t n, size_t a[]);
bool is_dim_to_split(size_t i, std::vector<size_t> dims);
drefine_status run_refine_poly_for_one_task(Network_t* net);
drefine_status run_milp_refine_with_milp_mark_input_split(Network_t* net);

#endif