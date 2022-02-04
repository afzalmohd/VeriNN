//#include "../../deeppoly/configuration.hh"
#include "../../deeppoly/deeppoly_driver.hh"
#include "../../deeppoly/deeppoly_configuration.hh"
#include "drefine_driver.hh"
#include "pullback.hh"
#include "decision_making.hh"
#include "milp_refine.hh"
#include "milp_mark.hh"
#include<cmath>
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
        //run_milp_refinement_with_pullback(net);
        run_path_split_with_pullback(net);
        //run_milp_refine_with_milp_mark(net);
        
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

void run_milp_refinement_with_pullback(Network_t* net){
    bool is_ce, is_verified = false;
    size_t loop_counter = 0;
    size_t upper_iter_limit = 15;
    while (true){
        is_ce = pull_back_full(net);
        if(is_ce){
            std::cout<<"Found counter example"<<std::endl;
            break;
        }
        else{
            is_verified = is_image_verified_by_milp(net);
            if(is_verified){
                std::cout<<"Image: "<<net->pred_label<<" verified!\n";
                break;
            }
        }
        loop_counter++;
        if(loop_counter >= upper_iter_limit){
            std::cout<<"Loop counter limit exceeded!\n";
            break;
        }
    }
}

void run_path_split_with_pullback(Network_t* net){
    bool is_ce = pull_back_full(net);
    if(is_ce){
        std::cout<<"Found counter example"<<std::endl;
    }
    else{
        std::vector<std::vector<Neuron_t*>> marked_vec;
        bool is_marked_neurons_added = marked_neurons_vector(net, marked_vec);
        assert(is_marked_neurons_added && "New neurons must added\n");
        bool is_path_available = set_marked_path(net, marked_vec, true);
        size_t counter = 1;
        size_t upper_limit = 100;
        bool is_verified = false;
        while (is_path_available && counter <= upper_limit){
            std::cout<<"Marked iteration: "<<counter<<std::endl;
            counter++;
            is_verified = run_deeppoly(net);
            if(is_verified){
                is_path_available = set_marked_path(net, marked_vec, false);
            }
            else{
                is_ce = pull_back_full(net);
                if(is_ce){
                    xt::xarray<double> res;
                    std::vector<double> vec;
                    std::cout<<"Found counter example"<<std::endl;
                    std::cout<<"[";
                    for(Neuron_t* nt : net->input_layer->neurons){
                        std::cout<<nt->back_prop_ub<<", ";
                        vec.push_back(nt->back_prop_ub);
                    }
                    std::cout<<"]"<<std::endl;
                    std::vector<size_t> shape = {net->input_dim};
                    res = xt::adapt(vec, shape);
                    net->forward_propgate_network(0, res);
                    auto pred_label = xt::argmax(net->layer_vec.back()->res);
                    std::cout<<"Pred label: "<<pred_label[0]<<std::endl;
                    break;
                }
                else{
                    is_marked_neurons_added = marked_neurons_vector(net, marked_vec);
                    is_path_available = set_marked_path(net, marked_vec, is_marked_neurons_added);
                }
            }
        }
        if(!is_path_available){
            std::cout<<"Image: "<<net->pred_label<<" verified!\n";
        }
        else if(!is_ce){
            std::cout<<"Loop count iteration exceed\n";
        }
    }
}

void run_milp_refine_with_milp_mark(Network_t* net){
    net->verified_out_dims.clear();
    net->counter_class_dim = net->actual_label;
    size_t loop_upper_bound = 10;
    size_t loop_counter = 0;
    while(loop_counter < loop_upper_bound){
        bool is_ce = run_milp_mark_with_milp_refine(net);
        std::cout<<"refinement iteration: "<<loop_counter<<std::endl;
        if(is_ce){
            std::cout<<"Found counter example!!"<<std::endl;
            break;
        }
        else{
            bool is_image_verified = is_image_verified_by_milp(net);
            if(is_image_verified){
                std::cout<<"Image verified!!"<<std::endl;
                break;
            }
        }
        loop_counter++;
    }
}
