#include "configuration.hh"
#include <iostream>

namespace Configuration{
    po::options_description desc("Options");
    po::variables_map vm;
    std::string default_root_path = "/home/u1411251/Documents/Phd/tools/VeriNN/deep_refine";
    std::string default_abs_out_file_path = default_root_path+"/benchmarks/fppolyForward.txt";
    std::string default_net_path = default_root_path+"/benchmarks/mnist_relu_3_50.tf";
    std::string default_dataset_path = default_root_path+"/benchmarks/mnist_test.csv";
    double default_epsilon = 0.03;

    std::string abs_out_file_path;
    std::string net_path;
    std::string dataset_path;
    double epsilon;

    int init_options(int num_of_params, char* params[]){
        try{
            desc.add_options()
            ("help,h", "produce help message")
            ("abs-out-file,aof", po::value<std::string>(&abs_out_file_path)->default_value(default_abs_out_file_path), "Output of abstraction to be refine")
            ("network", po::value<std::string>(&net_path)->default_value(default_net_path), "Neural network file")
            ("dataset", po::value<std::string>(&dataset_path)->default_value(default_dataset_path), "Dataset file in CSV form")
            ("epsilon", po::value<double>(&epsilon)->default_value(default_epsilon), "Value of image perturbation epsilon")
            ;
            po::store(po::parse_command_line(num_of_params, params, desc), vm);
            po::notify(vm);
            show_options(vm);
        }
        catch(std::exception& e){
            std::cout<<e.what()<<std::endl;
        }

        return 0;
    }

    int show_options(po::variables_map &vm){
        if (vm.count("help")) {
            std::cout << "Usage: options_description [options]\n";
            std::cout << desc;
            return 0;
        }
        
        if(vm.count("abs-out-file")){
            std::cout<<"Abstraction output file: "<<vm["abs-out-file"].as<std::string>() << std::endl;
        }

        if(vm.count("network")){
            std::cout<<"Network file: "<<vm["network"].as<std::string>() << std::endl;
        }

        if(vm.count("dataset")){
            std::cout<<"Dataset file: "<<vm["dataset"].as<std::string>() << std::endl;
        }

        if(vm.count("epsilon")){
            std::cout<<"Epsilon: "<<vm["epsilon"].as<double>() << std::endl;
        }

        return 0;
    }
    
}