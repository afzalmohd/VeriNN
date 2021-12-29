//#include "../../deeppoly/configuration.hh"
#include "../../deeppoly/deeppoly_driver.hh"
#include "drefine_driver.hh"
#include "configuration.hh"
#include<fstream>

int run_refine_poly(int num_args, char* params[]){
    int is_help = deeppoly_set_params(num_args, params);
    Configuration::init_options(num_args, params);
    if(is_help || Configuration::vm.count("help")){
        return 1;
    }

    Network_t* net = deeppoly_initialize_network();
    std::string image_str = get_image_str(Configuration::dataset_path, 1);
    deeppoly_parse_input_image_string(net, image_str);
    net->pred_label = execute_network(net);
    if(net->actual_label != net->pred_label){
        std::cout<<"Image (actual, pred_label): ("<<net->actual_label<<", "<<net->pred_label<<")"<<std::endl;
        return 1;
    }

    bool is_verified = run_deeppoly(net);
    if(is_verified){
        std::cout<<"Image: "<<net->pred_label<<" verified!\n";
    }
    else{
        std::cout<<"Image: "<<net->pred_label<<" not verified!\n";
    }


    
    return 0;
}

std::string get_image_str(std::string& image_path, size_t image_index){
    std::fstream newfile;
    newfile.open(image_path, std::ios::in);
    if(newfile.is_open()){
        std::string tp;
        size_t image_counter = 0;
        while (getline(newfile, tp)){
            if(tp != ""){
                if(image_counter == image_index){
                    return tp;
                }
                else{
                    image_counter++;
                }
            }
        }
    }
    assert(0 && "either empty image file or image_index out of bound");
    return "";
}
