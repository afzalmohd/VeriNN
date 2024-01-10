#ifndef __DEEPPOLY_CONFIGURATION_HH__
#define __DEEPPOLY_CONFIGURATION_HH__
#include <boost/program_options.hpp>
//#define RELU "ReLU"
//#define FC "FC"
#define VERBOSE true
#if VERBOSE
    #define IFVERBOSE(x) x
#else
    #define IFVERBOSE(x) (void*) 0
#endif

#define IMAGE_DELIMETER ','
#define NUM_TEST_IMAGES 30
#define MILP_WITH_MILP_LIMIT 2000
#define MILP_WITH_MILP_LIMIT_WITH_INPUT_SPLIT 3
#define NUM_GUROBI_THREAD 2
#define DIFF_TOLERANCE 1e-5
#define NUM_RANDOM_IMAGES 20
#define MAX_NUM_MARKED_NEURONS 5
#define MAX_INPUT_DIMS_TO_SPLIT 2
#define IS_LIGHT_WEIGHT_MARKED_ANALYSIS false
#define IS_LP_CONSTRAINTS false
#define IS_TOP_MIN_DIFF false
#define IS_CONF_CE true
#define CONFIDENCE_OF_CE 0.95
#define IS_TARGET_CE (IS_CONF_CE && false)
#define TARGET_CLASS 2

enum drefine_status {FAILED, DEEPPOLY_VERIFIED, VERIFIED, UNKNOWN};



namespace po = boost::program_options;

namespace Configuration_deeppoly{
    extern int init_options(int num_of_params, char* params[]);
    extern int show_options(po::variables_map &vm);
    extern po::options_description desc;
    extern po::variables_map vm;
    
    extern std::string default_root_path;
    extern std::string default_abs_out_file_path;
    extern std::string default_net_path;
    extern std::string default_dataset_path;
    extern double default_epsilon;
    extern std::string default_dataset;
    extern std::string default_tool;
    extern std::vector<std::string> dataset_vec;

    extern std::string abs_out_file_path;
    extern std::string net_path;
    extern std::string dataset_path;
    extern std::string dataset;
    extern size_t input_dim;
    extern double epsilon;
    extern bool is_small_ex;
    extern bool is_parallel;
    extern unsigned int num_thread;
    extern std::string result_file;
    extern size_t image_index;
    extern std::string tool;
    extern std::string vnnlib_prp_file_path;
    extern bool is_input_split;
    extern std::string bounds_path;

}


#endif