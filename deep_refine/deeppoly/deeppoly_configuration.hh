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

    extern std::string abs_out_file_path;
    extern std::string net_path;
    extern std::string dataset_path;
    extern std::string dataset;
    extern size_t input_dim;
    extern double epsilon;
    extern bool is_small_ex;
    extern bool is_parallel;
    extern unsigned int num_thread;
    extern bool is_unmarked_deeppoly;

}


#endif