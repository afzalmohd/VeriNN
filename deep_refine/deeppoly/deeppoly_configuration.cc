#include "deeppoly_configuration.hh"
#include<iostream>

namespace Configuration_deeppoly{
    po::options_description desc("Options");
    po::variables_map vm;
    std::string default_root_path = "/home/u1411251/Documents/Phd/tools";
    std::string default_net_path = default_root_path+"/networks/mnist_relu_3_50.tf";
    std::string default_dataset_path = default_root_path+"/dataset/mnist_test.csv";
    std::string default_dataset = "MNIST";
    double default_epsilon = 0.03;
    size_t input_dim;

    std::string net_path;
    std::string dataset_path;
    std::string dataset;
    double epsilon;
    bool is_small_ex;
    bool is_parallel;
    unsigned int num_thread;

    int init_options(int num_of_params, char* params[]){
        try{
            desc.add_options()
            ("help,h", "produce help message")
            ("network", po::value<std::string>(&net_path)->default_value(default_net_path), "Neural network file")
            ("dataset-file", po::value<std::string>(&dataset_path)->default_value(default_dataset_path), "Dataset file in CSV form")
            ("epsilon", po::value<double>(&epsilon)->default_value(default_epsilon), "Value of image perturbation epsilon")
            ("dataset", po::value<std::string>(&dataset)->default_value(default_dataset), "Types of image dataset")
            ("is-small-example,ise", po::value<bool>(&is_small_ex)->default_value(false), "Small example for testing")
            ("is-parallel", po::value<bool>(&is_parallel)->default_value(false), "Use parallelization")
            ("num-thread", po::value<unsigned int>(&num_thread)->default_value(4), "Number of cores in parallelization")
            ;
            po::store(po::parse_command_line(num_of_params, params, desc), vm);
            po::notify(vm);
            show_options(vm);

            if(dataset == "MNIST"){
                input_dim = 784;
            }
            if(is_small_ex){
                input_dim = 2;
            }


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
            std::cout<<"Dataset type: "<<vm["dataset"].as<std::string>() << std::endl;
        }

        if(vm.count("dataset-file")){
            std::cout<<"Dataset file: "<<vm["dataset-file"].as<std::string>() << std::endl;
        }

        if(vm.count("epsilon")){
            std::cout<<"Epsilon: "<<vm["epsilon"].as<double>() << std::endl;
        }

        if(vm.count("is-small-example")){
            if(vm["is-small-example"].as<bool>()){
                std::cout<<"Is small test case: True"<< std::endl;
            }
            else{
                std::cout<<"Is small test case: False" << std::endl;
            }
        }

        return 0;
    }
    
}