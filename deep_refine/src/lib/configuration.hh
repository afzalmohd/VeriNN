#ifndef _CONFIGURATION_HH_
#define _CONFIGURATION_HH_
#include <boost/program_options.hpp>
namespace po = boost::program_options;

namespace Configuration{
    extern int init_options(int num_of_params, char* params[]);
    extern int show_options(po::variables_map &vm);
    extern po::options_description desc;
    extern po::variables_map vm;
    
    extern std::string default_root_path;
    extern std::string default_abs_out_file_path;
    extern std::string default_net_path;
    extern std::string default_dataset_path;
    extern double default_epsilon;


    extern std::string abs_out_file_path;
    extern std::string net_path;
    extern std::string dataset_path;
    extern double epsilon;
}

#endif