#ifndef __DREFINE_DRIVER__
#define __DREFINE_DRIVER__
#include<string>
std::string get_image_str(std::string& image_path, size_t image_index);
void run_milp_refinement_with_pullback(Network_t* net);
void run_path_split_with_pullback(Network_t* net);
void run_milp_refine_with_milp_mark(Network_t* net);
#endif