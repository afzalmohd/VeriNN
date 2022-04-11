#ifndef __DREFINE_DRIVER__
#define __DREFINE_DRIVER__
#include<string>
int run_refine_poly_for_one_image(Network_t* net, size_t image_index, std::chrono::_V2::system_clock::time_point start_time);
std::string get_image_str(std::string& image_path, size_t image_index);
void run_milp_refinement_with_pullback(Network_t* net, size_t image_index, std::chrono::_V2::system_clock::time_point start_time);
void run_path_split_with_pullback(Network_t* net, size_t image_index, std::chrono::_V2::system_clock::time_point start_time);
void run_milp_refine_with_milp_mark(Network_t* net, size_t image_index, std::chrono::_V2::system_clock::time_point start_time);
void create_ce_and_run_nn(Network_t* net);
void write_to_file(std::string& file_path, std::string& s);
void print_failed_string(Network_t* net, size_t image_index, size_t loop_counter, std::chrono::_V2::system_clock::time_point start_time);
void print_verified_string(Network_t* net, size_t image_index, size_t loop_counter, std::chrono::_V2::system_clock::time_point start_time);
void print_unknown_string(Network_t* net, size_t image_index, size_t loop_counter, std::chrono::_V2::system_clock::time_point start_time);
bool is_real_ce_mnist_cifar10(Network_t* net);
std::string get_absolute_file_name_from_path(std::string & path);
bool is_valid_dataset();
void set_stds_means(Network_t* net);
void normalize_input_image(Network_t* net); //use for the first time
void normalize_image(Network_t* net); //use in between of the execution of the tool
void denormalize_image(Network_t* net);
#endif