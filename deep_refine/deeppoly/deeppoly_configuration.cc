#include "deeppoly_configuration.hh"
#include<iostream>


namespace Global_vars{
    std::vector<Neuron_t*> new_marked_nts;    
    size_t iter_counts = 0; //to count the number cegar iterations
    size_t sub_prob_counts = 0; // to count the number of sub problems when input_split on
    size_t num_marked_neurons = 0; // to count the number of marked neurons
    std::chrono::duration<double> marking_time = std::chrono::seconds(0);
    std::chrono::duration<double> refinement_time = std::chrono::seconds(0);
    double orig_im_conf = 0;
    double ce_im_conf = 0;
}

namespace Configuration_deeppoly{
    po::options_description desc("Options");
    po::variables_map vm;
    std::string default_root_path = "/home/u1411251/Documents/tools/VeriNN/deep_refine";
    std::string default_net_path = "/home/u1411251/Documents/tools/networks/tf/mnist/mnist_relu_3_50.tf";
    std::string default_dataset_path = default_root_path+"/benchmarks/dataset/mnist/mnist_test.csv";
    std::string default_result_file = default_root_path+"/outfiles/result.txt";
    std::string default_dataset = "MNIST";
    std::string default_tool = "drefine";
    double default_epsilon = 0.03;
    std::vector<std::string> dataset_vec = {"MNIST","CIFAR10","ACASXU"};
    size_t input_dim = 784;
    bool is_reset_marked_nts = IS_RESET_MARK_FOR_EACH_LABLE;

    std::string net_path;
    std::string dataset_path;
    std::string dataset;
    double epsilon;
    bool is_small_ex;
    bool is_parallel;
    unsigned int num_thread;
    std::string result_file;
    size_t image_index;
    std::string tool;
    std::string vnnlib_prp_file_path;
    bool is_input_split;
    std::string bounds_path;
    bool is_conf_ce;
    bool is_target_ce;
    double conf_of_ce;
    bool is_concurrent = IS_CONCURRENT_RUN;
    

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
            ("result-file", po::value<std::string>(&result_file)->default_value(default_result_file), "result file")
            ("image-index", po::value<size_t>(&image_index)->default_value(1), "Image index to be verify")
            ("tool", po::value<std::string>(&tool)->default_value(default_tool), "tool name drefine/deeppoly")
            ("vnnlib-prp-file,vnnlib", po::value<std::string>(&vnnlib_prp_file_path)->default_value(""), "vnnlib prp file path")
            ("is-input-split", po::value<bool>(&is_input_split)->default_value(false), "run with heuristic input space split")
            ("bounds-path", po::value<std::string>(&bounds_path)->default_value(""), "external bounds")
            ("is-conf-ce", po::value<bool>(&is_conf_ce)->default_value(false), "is coounter example with high confidence")
            ("is-target-ce", po::value<bool>(&is_target_ce)->default_value(false), "is coounter example with target class")
            ("ce-conf", po::value<double>(&conf_of_ce)->default_value(0.90), "counter class confidence")
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

        if(vm.count("is-input-split")){
            if(vm["is-input-split"].as<bool>()){
                std::cout<<"Is input split: True"<< std::endl;
            }
            else{
                std::cout<<"Is input split: False"<< std::endl;
            }
        }

        if(vm.count("is-conf-ce")){
            std::cout<<"Is ce with high confidence: "<<vm["is-conf-ce"].as<bool>()<<std::endl;
        }

        if(vm.count("ce-conf")){
            std::cout<<"CE conf: "<<vm["ce-conf"].as<double>()<<std::endl;
        }

        return 0;
    }
    
}