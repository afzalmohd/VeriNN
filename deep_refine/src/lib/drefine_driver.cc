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
#include<iostream>
#include<chrono>

void testing(Network_t* net){
    std::vector<double> vec;
    vec.reserve(net->layer_vec[0]->dims);
    for(size_t i=0; i<net->layer_vec[0]->dims; i++){
        vec.push_back(0);
    }
    std::vector<size_t> shape = {net->layer_vec[0]->dims};
    xt::xarray<double> res = xt::adapt(vec, shape);
    std::cout<<res<<std::endl;
    net->forward_propgate_network(1, res);
    auto pred_label = xt::argmax(net->layer_vec.back()->res);
    std::cout<<"Pred label: "<<pred_label[0]<<" , "<<net->layer_vec.back()->res<<std::endl;
    std::vector<GRBVar> var_vector;
    var_vector.reserve(net->input_layer->dims);
    GRBModel model = create_grb_env_and_model();
    for(Neuron_t* nt : net->input_layer->neurons){
        std::string var_str = "x"+std::to_string(nt->neuron_index);
        GRBVar x = model.addVar(-nt->lb, nt->ub, 0.0, GRB_CONTINUOUS, var_str);
        var_vector.push_back(x);
    }

    for(size_t i=0; i<net->layer_vec[0]->dims; i++){
        Neuron_t* nt = net->layer_vec[0]->neurons[i];
        Expr_t* expr = nt->uexpr;
        GRBLinExpr grb_expr;
        grb_expr.addTerms(&expr->coeff_sup[0], &var_vector[0], var_vector.size());
        grb_expr += expr->cst_sup;
        model.addConstr(grb_expr==0);
    }

    model.optimize();
    int status = model.get(GRB_IntAttr_Status);
    if(status == GRB_OPTIMAL){
        std::cout<<"Found counter example!!"<<std::endl;
        std::vector<double> vec;
        xt::xarray<double> res;
        for(size_t i=0; i< net->input_dim; i++){
            double val = var_vector[i].get(GRB_DoubleAttr_X);
            vec.push_back(val);
        }
        std::vector<size_t> shape = {net->input_dim};
        res = xt::adapt(vec, shape);
        net->forward_propgate_network(0, res);
        auto pred_label = xt::argmax(net->layer_vec.back()->res);
        std::cout<<res<<std::endl;
        std::cout<<"Pred label: "<<pred_label[0]<<" , "<<net->layer_vec.back()->res<<std::endl;
    }
    else{
        std::cout<<"Not found!!"<<status<<std::endl;
    }
}

int run_refine_poly(int num_args, char* params[]){
    int is_help = deeppoly_set_params(num_args, params);
    if(is_help){
        return 1;
    }
    //size_t num_images = NUM_TEST_IMAGES;
    Network_t* net = deeppoly_initialize_network();
    // for(size_t i = 1; i<num_images+1; i++){
    //     auto start_time =  std::chrono::high_resolution_clock::now();
    //     if( i > 1){
    //         break;
    //     }
    //     run_refine_poly_for_one_image(net, i, start_time);
    // }

    auto start_time =  std::chrono::high_resolution_clock::now();
    run_refine_poly_for_one_image(net, Configuration_deeppoly::image_index, start_time);


    
    return 0;
}

void run_refine_poly_for_one_image(Network_t* net, size_t image_index, std::chrono::_V2::system_clock::time_point start_time){
    std::cout<<"Image index: "<<image_index<<std::endl;
    std::string image_str = get_image_str(Configuration_deeppoly::dataset_path, image_index);
    deeppoly_parse_input_image_string(net, image_str);
    net->pred_label = execute_network(net);
    if(net->actual_label != net->pred_label){
        std::string str = "Image,"+std::to_string(image_index)+", (actual pred_label),("+std::to_string(net->actual_label)+" "+std::to_string(net->pred_label)+")";
        std::cout<<str<<std::endl;
        write_to_file(Configuration_deeppoly::result_file, str);
        return;
    }
    Configuration_deeppoly::is_unmarked_deeppoly = true;
    bool is_verified = run_deeppoly(net);
    Configuration_deeppoly::is_unmarked_deeppoly = false;
    //testing(net);
    //return 0;
    if(is_verified){
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        std::string str = "Image,"+std::to_string(image_index)+",label,"+std::to_string(net->pred_label)+",verified,deeppoly,time,"+std::to_string(duration.count());
        std::cout<<str<<std::endl;
        write_to_file(Configuration_deeppoly::result_file, str);
    }
    else{
        std::cout<<"Image: "<<net->pred_label<<" not verified!\n";
        if(Configuration_deeppoly::is_milp_based_mark && Configuration_deeppoly::is_milp_based_refine){
            run_milp_refine_with_milp_mark(net, image_index, start_time);
        }
        else if(!Configuration_deeppoly::is_milp_based_mark && !Configuration_deeppoly::is_milp_based_refine){
            run_path_split_with_pullback(net, image_index, start_time);
        }
        else if(!Configuration_deeppoly::is_milp_based_mark && Configuration_deeppoly::is_milp_based_refine){
            std::cout<<"Pull back with milp based refinement not workable"<<std::endl;
            return;
        }
        else if(Configuration_deeppoly::is_milp_based_mark && !Configuration_deeppoly::is_milp_based_refine){
            std::cout<<"Optimization marked with path spliting not yet implemented"<<std::endl;
            return;
        }
        
    }
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

void run_milp_refinement_with_pullback(Network_t* net, size_t image_index, std::chrono::_V2::system_clock::time_point start_time){
    bool is_ce, is_verified = false;
    size_t loop_counter = 0;
    size_t upper_iter_limit = PULL_BACK_WITH_MILP_LIMIT;
    while (true){
        is_ce = pull_back_full(net);
        if(is_ce){
            print_failed_string(net, image_index, loop_counter, start_time);
            break;
        }
        else{
            is_verified = is_image_verified_by_milp(net);
            if(is_verified){
                print_verified_string(net, image_index, loop_counter, start_time);
                break;
            }
        }
        loop_counter++;
        if(loop_counter >= upper_iter_limit){
            print_unknown_string(net, image_index, loop_counter, start_time);
            break;
        }
    }
}

void run_path_split_with_pullback(Network_t* net, size_t image_index, std::chrono::_V2::system_clock::time_point start_time){
    bool is_ce = pull_back_full(net);
    size_t counter = 1;
    if(is_ce){
        print_failed_string(net, image_index, counter, start_time);
    }
    else{
        std::vector<std::vector<Neuron_t*>> marked_vec;
        bool is_marked_neurons_added = marked_neurons_vector(net, marked_vec);
        assert(is_marked_neurons_added && "New neurons must added\n");
        bool is_path_available = set_marked_path(net, marked_vec, true);
        size_t upper_limit = PULL_BACK_WITH_PATH_SPLIT;
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
                    print_failed_string(net, image_index, counter, start_time);
                    break;
                }
                else{
                    is_marked_neurons_added = marked_neurons_vector(net, marked_vec);
                    is_path_available = set_marked_path(net, marked_vec, is_marked_neurons_added);
                }
            }
        }
        if(!is_path_available){
            print_verified_string(net, image_index, counter, start_time);
        }
        else if(!is_ce){
            print_unknown_string(net, image_index, counter, start_time);
        }
    }
}

void run_milp_refine_with_milp_mark(Network_t* net, size_t image_index, std::chrono::_V2::system_clock::time_point start_time){
    //net->verified_out_dims.clear();
    net->counter_class_dim = net->actual_label;
    size_t loop_upper_bound = MILP_WITH_MILP_LIMIT;
    size_t loop_counter = 0;
    while(loop_counter < loop_upper_bound){
        bool is_ce = run_milp_mark_with_milp_refine(net);
        std::cout<<"refinement iteration: "<<loop_counter<<std::endl;
        if(is_ce){
            print_failed_string(net, image_index, loop_counter, start_time);
            return;
        }
        else{
            bool is_image_verified = is_image_verified_by_milp(net);
            if(is_image_verified){
                print_verified_string(net, image_index, loop_counter, start_time);
                return;
            }
        }
        loop_counter++;
    }
    print_unknown_string(net, image_index, loop_counter, start_time);

}

void create_ce_and_run_nn(Network_t* net){
    xt::xarray<double> res;
    std::vector<double> vec;
    
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
    std::cout<<"Pred label: "<<pred_label[0]<<" , "<<net->layer_vec.back()->res<<std::endl;
}

void write_to_file(std::string& file_path, std::string& s){
    std::ofstream my_file;
    my_file.open(file_path, std::ios::app);
    my_file<<s<<std::endl;
    my_file.close();
}

void print_failed_string(Network_t* net, size_t image_index, size_t loop_counter, std::chrono::_V2::system_clock::time_point start_time){
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::string str = "Image,"+std::to_string(image_index)+",label,"+std::to_string(net->pred_label)+",failed,count,"+std::to_string(loop_counter)+",time,"+std::to_string(duration.count());
    write_to_file(Configuration_deeppoly::result_file, str);
    std::cout<<str<<std::endl;
}

void print_verified_string(Network_t* net, size_t image_index, size_t loop_counter, std::chrono::_V2::system_clock::time_point start_time){
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::string str = "Image,"+std::to_string(image_index)+",label,"+std::to_string(net->pred_label)+",verified,count,"+std::to_string(loop_counter)+",time,"+std::to_string(duration.count());
    write_to_file(Configuration_deeppoly::result_file, str);
    std::cout<<str<<std::endl;
}

void print_unknown_string(Network_t* net, size_t image_index, size_t loop_counter, std::chrono::_V2::system_clock::time_point start_time){
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::string str = "Image,"+std::to_string(image_index)+",label,"+std::to_string(net->pred_label)+",unknown,count,"+std::to_string(loop_counter)+",time,"+std::to_string(duration.count());
    write_to_file(Configuration_deeppoly::result_file, str);
    std::cout<<str<<std::endl;
}


