#include "deeppoly_configuration.hh"
#include<iostream>

namespace Configuration_deeppoly{
    po::options_description desc("Options");
    po::variables_map vm;
    std::string default_root_path = "/home/u1411251/Documents/Phd/tools/VeriNN/deep_refine";
    std::string default_net_path = default_root_path+"/benchmarks/networks/tf/mnist_relu_3_50.tf";
    std::string default_dataset_path = default_root_path+"/benchmarks/dataset/mnist/mnist_test.csv";
    std::string default_result_file = default_root_path+"/outfiles/result.txt";
    std::string default_dataset = "MNIST";
    std::string default_tool = "drefine";
    double default_epsilon = 0.03;
    std::vector<std::string> dataset_vec = {"MNIST","CIFAR10","ACASXU"};
    size_t input_dim = 784;

    std::string net_path;
    std::string dataset_path;
    std::string dataset;
    double epsilon;
    bool is_small_ex;
    bool is_parallel;
    unsigned int num_thread;
    bool is_unmarked_deeppoly = true;
    bool is_milp_based_refine;
    bool is_milp_based_mark;
    std::string result_file;
    size_t image_index;
    std::string tool;

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
            ("is-milp-refine", po::value<bool>(&is_milp_based_refine)->default_value(true), "Is milp based refinement")
            ("is-optimization-mark", po::value<bool>(&is_milp_based_mark)->default_value(true), "Is optimizationan based refinement")
            ("result-file", po::value<std::string>(&result_file)->default_value(default_result_file), "result file")
            ("image-index", po::value<size_t>(&image_index)->default_value(1), "Image index to be verify")
            ("tool", po::value<std::string>(&tool)->default_value(default_tool), "tool name drefine/deeppoly")
            ;
            po::store(po::parse_command_line(num_of_params, params, desc), vm);
            po::notify(vm);
            show_options(vm);

            if(is_small_ex){
                input_dim = 2;
            }

            if(dataset == "MNIST"){
                input_dim = 784;
            }
            else if(dataset == "CIFAR10"){
                input_dim = 3072;
            }
            else if(dataset == "ACASXU"){
                input_dim = 5;
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

        if(vm.count("tool")){
            std::cout<<"Tool: "<<vm["tool"].as<std::string>()<<std::endl;
        }

        if(vm.count("is-parallel")){
            if(vm["is-parallel"].as<bool>()){
                std::cout<<"Parallization: on" << std::endl;
            }
            else{
                std::cout<<"Parallization: off" << std::endl;
            }
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