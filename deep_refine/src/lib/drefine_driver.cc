//#include "../../deeppoly/configuration.hh"
#include "../../deeppoly/deeppoly_driver.hh"
#include "../../deeppoly/deeppoly_configuration.hh"
#include "drefine_driver.hh"
#include "pullback.hh"
#include "decision_making.hh"
#include<fstream>

int run_refine_poly(int num_args, char* params[]){
    int is_help = deeppoly_set_params(num_args, params);
    if(is_help){
        return 1;
    }

    Network_t* net = deeppoly_initialize_network();
    std::string image_str = get_image_str(Configuration_deeppoly::dataset_path, 1);
    deeppoly_parse_input_image_string(net, image_str);
    net->pred_label = execute_network(net);
    if(net->actual_label != net->pred_label){
        std::cout<<"Image (actual, pred_label): ("<<net->actual_label<<", "<<net->pred_label<<")"<<std::endl;
        return 1;
    }
    Configuration_deeppoly::is_unmarked_deeppoly = true;
    bool is_verified = run_deeppoly(net);
    Configuration_deeppoly::is_unmarked_deeppoly = false;
    if(is_verified){
        std::cout<<"Image: "<<net->pred_label<<" verified!\n";
    }
    else{
        std::cout<<"Image: "<<net->pred_label<<" not verified!\n";
        bool is_ce = pull_back_full(net);
        if(is_ce){
            std::cout<<"Found counter example"<<std::endl;
            return 0;
        }
        else{
            std::vector<std::vector<Neuron_t*>> marked_vec;
            bool is_marked_neurons_added = marked_neurons_vector(net, marked_vec);
            assert(is_marked_neurons_added && "New neurons must added\n");
            bool is_path_available = set_marked_path(net, marked_vec, true);
            size_t counter = 1;
            size_t upper_limit = 100;
            while (is_path_available && counter <= upper_limit){
                std::cout<<"Marked iteration: "<<counter<<std::endl;
                counter++;
                is_verified = run_deeppoly(net);
                printf("Check..1\n");
                if(is_verified){
                    is_path_available = set_marked_path(net, marked_vec, false);
                }
                else{
                    is_ce = pull_back_full(net);
                    if(is_ce){
                        std::cout<<"Found counter example"<<std::endl;
                        std::cout<<"[";
                        for(Neuron_t* nt : net->input_layer->neurons){
                            std::cout<<nt->back_prop_ub<<", ";
                        }
                        std::cout<<"]"<<std::endl;
                        return 0;
                    }
                    else{
                        is_marked_neurons_added = marked_neurons_vector(net, marked_vec);
                        is_path_available = set_marked_path(net, marked_vec, is_marked_neurons_added);
                    }
                }
            }

            std::cout<<"Image: "<<net->pred_label<<" verified!\n";
        }
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