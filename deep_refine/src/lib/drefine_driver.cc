//#include "../../deeppoly/configuration.hh"
#include "../../deeppoly/deeppoly_driver.hh"
#include "../../deeppoly/deeppoly_configuration.hh"
#include "../../deeppoly/helper.hh"
#include "../../deeppoly/vnnlib.hh"
#include "drefine_driver.hh"
#include "pullback.hh"
#include "decision_making.hh"
#include "milp_refine.hh"
#include "milp_mark.hh"
#include<cmath>
#include<fstream>
#include<iostream>
#include<chrono>

int run_refine_poly(int num_args, char* params[]){
    int is_help = deeppoly_set_params(num_args, params);
    if(is_help || (!is_valid_dataset())){
        return 1;
    }

    if(Configuration_deeppoly::vnnlib_prp_file_path != ""){
        VnnLib_t* verinn_lib = parse_vnnlib(Configuration_deeppoly::vnnlib_prp_file_path);
        Network_t* net = deeppoly_initialize_network();
        net->vnn_lib = verinn_lib;
        int status = run_drefine_vnnlib(net);
        return status;
    }
    Network_t* net = deeppoly_initialize_network();
    set_stds_means(net);

    auto start_time =  std::chrono::high_resolution_clock::now();
    bool is_same_label = is_actual_and_pred_label_same(net, Configuration_deeppoly::image_index);
    if(!is_same_label){
        return 0;
    }
    run_refine_poly_for_one_task(net, start_time);


    
    return 0;
}

bool is_actual_and_pred_label_same(Network_t* net, size_t image_index){
    std::cout<<"Image index: "<<image_index<<std::endl;
    std::string image_str = get_image_str(Configuration_deeppoly::dataset_path, image_index);
    deeppoly_parse_input_image_string(net, image_str);
    normalize_input_image(net);
    net->pred_label = execute_network(net);
    for(size_t i=0; i<net->output_dim; i++){
        std::cout<<net->layer_vec.back()->res[i]<<" ";
    }
    std::cout<<std::endl;
    if(net->actual_label != net->pred_label){
        std::string base_net_name = get_absolute_file_name_from_path(Configuration_deeppoly::net_path);
        std::string str = base_net_name+","+std::to_string(Configuration_deeppoly::epsilon)+","+std::to_string(image_index)+","+std::to_string(net->actual_label)+" "+std::to_string(net->pred_label)+",null,wrong_pred,network,0,0,0";
        write_to_file(Configuration_deeppoly::result_file, str);
        std::cout<<str<<std::endl;
        return false;
        
    }
    return true;
}

int run_refine_poly_for_one_task(Network_t* net, std::chrono::_V2::system_clock::time_point start_time){
    size_t image_index = Configuration_deeppoly::image_index;
    // bool is_ce = is_ce_cheap_check(net);
    // if(is_ce){
    //     std::cout<<"Got counter example!!!"<<std::endl;
    //     print_status_string(net, 2, "pre-check", image_index, 0, start_time);
    //     return 0;
    // }

    Configuration_deeppoly::is_unmarked_deeppoly = true;
    std::string tool_name = Configuration_deeppoly::tool;
    Configuration_deeppoly::tool = "deeppoly";
    bool is_verified = run_deeppoly(net);
    Configuration_deeppoly::tool = tool_name;
    Configuration_deeppoly::is_unmarked_deeppoly = false;
    if(is_verified){
        print_status_string(net, 1, "deeppoly", image_index, 0, start_time);
        return 1;
    }
    else{
        std::cout<<"Image: "<<net->pred_label<<" not verified!\n";
        if(Configuration_deeppoly::tool == "deeppoly"){
            print_status_string(net, 2, "deeppoly", image_index, 0, start_time);
            return 0;
        }

        is_verified = is_image_verified_deeppoly(net);
        if(is_verified){
            print_status_string(net, 1, "drefine", image_index, 0, start_time);
            return 1;
        }

        if(Configuration_deeppoly::is_milp_based_mark && Configuration_deeppoly::is_milp_based_refine){
            run_milp_refine_with_milp_mark(net, image_index, start_time);
        }
        else if(!Configuration_deeppoly::is_milp_based_mark && !Configuration_deeppoly::is_milp_based_refine){
            run_path_split_with_pullback(net, image_index, start_time);
        }
        else if(!Configuration_deeppoly::is_milp_based_mark && Configuration_deeppoly::is_milp_based_refine){
            std::cout<<"Pull back with milp based refinement not workable"<<std::endl;
            return 0;
        }
        else if(Configuration_deeppoly::is_milp_based_mark && !Configuration_deeppoly::is_milp_based_refine){
            std::cout<<"Optimization marked with path spliting not yet implemented"<<std::endl;
            return 0;
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

int run_milp_refinement_with_pullback(Network_t* net, size_t image_index, std::chrono::_V2::system_clock::time_point start_time){
    bool is_ce, is_verified = false;
    size_t loop_counter = 1;
    size_t upper_iter_limit = PULL_BACK_WITH_MILP_LIMIT;
    int status;
    while (true){
        is_ce = pull_back_full(net);
        if(is_ce){
            bool is_real_ce = is_real_ce_mnist_cifar10(net);
            if(is_real_ce){
                std::cout<<"Found counter example!!"<<std::endl;
                print_status_string(net, 0, "drefine", image_index, loop_counter, start_time);
                is_ce = true;
                status = Failed;
            }
            else{ //status unknown
                print_status_string(net, 2, "drefine", image_index, loop_counter, start_time);
                status = Unknown;
            }
            break;
        }
        else{
            is_verified = is_image_verified_by_milp(net);
            if(is_verified){
                print_status_string(net, 1, "drefine", image_index, loop_counter, start_time);
                status = Verified;
                break;
            }
        }
        loop_counter++;
        if(loop_counter >= upper_iter_limit){ //status unknown
            print_status_string(net, 2, "drefine", image_index, loop_counter, start_time);
            status = Unknown;
            break;
        }
    }
    return status;
}

int run_path_split_with_pullback(Network_t* net, size_t image_index, std::chrono::_V2::system_clock::time_point start_time){
    int status;
    bool is_ce = pull_back_full(net);
    size_t counter = 1;
    if(is_ce){
        bool is_real_ce = is_real_ce_mnist_cifar10(net);
        if(is_real_ce){
            std::cout<<"Found counter example!!"<<std::endl;
            print_status_string(net, 0, "drefine", image_index, counter, start_time);
            is_ce = true;
            status = Failed;
        }
        else{
           print_status_string(net, 2, "drefine", image_index, counter, start_time);
            status = Unknown;
        }
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
                    bool is_real_ce = is_real_ce_mnist_cifar10(net);
                    if(is_real_ce){
                        std::cout<<"Found counter example!!"<<std::endl;
                        print_status_string(net, 0, "drefine", image_index, counter, start_time);
                        is_ce = true;
                        status = Failed;
                    }
                    else{
                        print_status_string(net, 2, "drefine", image_index, counter, start_time);
                        status = Unknown;
                    }
                    break;
                }
                else{
                    is_marked_neurons_added = marked_neurons_vector(net, marked_vec);
                    is_path_available = set_marked_path(net, marked_vec, is_marked_neurons_added);
                }
            }
        }
        if(!is_path_available){
            print_status_string(net, 1, "drefine", image_index, counter, start_time);
            status = Verified;
        }
        else if(!is_ce){
            print_status_string(net, 2, "drefine", image_index, counter, start_time);
            status = Unknown;
        }
    }

    return status;
}

int run_milp_refine_with_milp_mark(Network_t* net, size_t image_index, std::chrono::_V2::system_clock::time_point start_time){
    //net->verified_out_dims.clear();
    int status;
    net->counter_class_dim = net->actual_label;
    size_t loop_upper_bound = MILP_WITH_MILP_LIMIT;
    size_t loop_counter = 0;
    bool is_bound_exceeded = true;
    while(loop_counter < loop_upper_bound){
        bool is_ce = run_milp_mark_with_milp_refine(net);
        std::cout<<"refinement iteration: "<<loop_counter<<std::endl;
        if(is_ce){
            bool is_real_ce = is_real_ce_mnist_cifar10(net);
            if(is_real_ce){
                std::cout<<"Found counter example!!"<<std::endl;
                print_status_string(net, 0, "drefine", image_index, loop_counter, start_time);
                status = Failed;
            }
            else{//unknown
                print_status_string(net, 2, "drefine", image_index, loop_counter, start_time);
                status = Unknown;
            }
            is_bound_exceeded = false;
            break;
        }
        else{
            bool is_image_verified = is_image_verified_by_milp(net);
            if(is_image_verified){
                print_status_string(net, 1, "drefine", image_index, loop_counter, start_time);
                is_bound_exceeded = false;
                status = Verified;
                break;
            }
        }
        loop_counter++;
    }
    if(is_bound_exceeded){ //unknown
        print_status_string(net, 2, "drefine", image_index, loop_counter, start_time);
        status = Unknown;
    }
    return status;
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

bool is_real_ce_mnist_cifar10(Network_t* net){
    denormalize_image(net);
    if(Configuration_deeppoly::dataset == "MNIST" || Configuration_deeppoly::dataset == "CIFAR10"){
        net->input_layer->res = net->input_layer->res * 255;
        net->input_layer->res = xt::round(net->input_layer->res);
        net->input_layer->res = net->input_layer->res/255;
    }
    normalize_image(net);
    net->forward_propgate_network(0, net->input_layer->res);
    auto pred_label = xt::argmax(net->layer_vec.back()->res);
    net->pred_label = pred_label[0];
    if(net->actual_label != net->pred_label){
        return true;
    }
    else{
        return false;
    }
}

void write_to_file(std::string& file_path, std::string& s){
    std::ofstream my_file;
    my_file.open(file_path, std::ios::app);
    if(my_file.is_open()){
        my_file<<s<<std::endl;
        my_file.close();
    }
    else{
        assert(0 && "result file could not open\n");
    }
}

void print_status_string(Network_t* net, size_t tool_status, std::string tool_name, size_t image_index, size_t loop_counter, std::chrono::_V2::system_clock::time_point start_time){
    auto end_time = std::chrono::high_resolution_clock::now();
    std::string status_string;
    if(tool_status == 0){
        status_string = "failed";
    }
    else if(tool_status == 1){
        status_string = "verified";
    }
    else{
        status_string = "unknown";
    }
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::string base_net_name = get_absolute_file_name_from_path(Configuration_deeppoly::net_path);
    std::string base_prp_name = get_absolute_file_name_from_path(Configuration_deeppoly::vnnlib_prp_file_path);
    size_t num_marked_nt = num_marked_neurons(net);
    if(base_prp_name == ""){
        base_prp_name = "null";
    }
    std::string str = base_net_name+","+std::to_string(Configuration_deeppoly::epsilon)+","+std::to_string(image_index)+","+std::to_string(net->pred_label)+","+base_prp_name+","+status_string+","+tool_name+","+std::to_string(loop_counter)+","+std::to_string(num_marked_nt)+","+std::to_string(duration.count());
    write_to_file(Configuration_deeppoly::result_file, str);
    std::cout<<str<<std::endl;
}

// void print_failed_string(Network_t* net, std::string tool_name, size_t image_index, size_t loop_counter, std::chrono::_V2::system_clock::time_point start_time){
//     auto end_time = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
//     std::string base_net_name = get_absolute_file_name_from_path(Configuration_deeppoly::net_path);
//     std::string base_prp_name = get_absolute_file_name_from_path(Configuration_deeppoly::vnnlib_prp_file_path);
//     size_t num_marked_nt = num_marked_neurons(net);
//     if(base_prp_name == ""){
//         base_prp_name = "null";
//     }
//     std::string str = base_net_name+","+std::to_string(Configuration_deeppoly::epsilon)+","+std::to_string(image_index)+","+std::to_string(net->pred_label)+","+base_prp_name+",failed,"+tool_name+","+std::to_string(loop_counter)+","+std::to_string(num_marked_nt)+","+std::to_string(duration.count());
//     write_to_file(Configuration_deeppoly::result_file, str);
//     std::cout<<str<<std::endl;
// }

// void print_verified_string(Network_t* net, std::string tool_name, size_t image_index, size_t loop_counter, std::chrono::_V2::system_clock::time_point start_time){
//     auto end_time = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
//     std::string base_net_name = get_absolute_file_name_from_path(Configuration_deeppoly::net_path);
//     std::string base_prp_name = get_absolute_file_name_from_path(Configuration_deeppoly::vnnlib_prp_file_path);
//     size_t num_marked_nt = num_marked_neurons(net);
//     if(base_prp_name == ""){
//         base_prp_name = "null";
//     }
//     std::string str = base_net_name+","+std::to_string(Configuration_deeppoly::epsilon)+","+std::to_string(image_index)+","+std::to_string(net->pred_label)+","+base_prp_name+",verified,"+tool_name+","+std::to_string(loop_counter)+","+std::to_string(num_marked_nt)+","+std::to_string(duration.count());
//     write_to_file(Configuration_deeppoly::result_file, str);
//     std::cout<<str<<std::endl;
// }

// void print_unknown_string(Network_t* net, std::string tool_name, size_t image_index, size_t loop_counter, std::chrono::_V2::system_clock::time_point start_time){
//     auto end_time = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
//     std::string base_net_name = get_absolute_file_name_from_path(Configuration_deeppoly::net_path);
//     std::string base_prp_name = get_absolute_file_name_from_path(Configuration_deeppoly::vnnlib_prp_file_path);
//     size_t num_marked_nt = num_marked_neurons(net);
//     if(base_prp_name == ""){
//         base_prp_name = "null";
//     }
//     std::string str = base_net_name+","+std::to_string(Configuration_deeppoly::epsilon)+","+std::to_string(image_index)+","+std::to_string(net->pred_label)+","+base_prp_name+",unknown,"+tool_name+","+std::to_string(loop_counter)+","+std::to_string(num_marked_nt)+","+std::to_string(duration.count());
//     write_to_file(Configuration_deeppoly::result_file, str);
//     std::cout<<str<<std::endl;
// }

std::string get_absolute_file_name_from_path(std::string & path){
    std::string base_net_name = path.substr(path.find_last_of("/")+1);
    return base_net_name;
}

size_t num_marked_neurons(Network_t* net){
    size_t num_marked_neurons = 0;
    for(Layer_t* layer : net->layer_vec){
        for(Neuron_t* nt : layer->neurons){
            if(nt->is_marked){
                num_marked_neurons += 1;
            }
        }
    }
    return num_marked_neurons;
}

bool is_valid_dataset(){
    std::vector<std::string> dataset_vec = Configuration_deeppoly::dataset_vec;
    bool is_valid_dataset = false;
    for(auto& dt:dataset_vec){
        if(Configuration_deeppoly::dataset == dt){
            is_valid_dataset = true;
            break;
        }
    }
    if(!is_valid_dataset){
        std::string valid_dts = dataset_vec[0];
        for(size_t i=1; i<dataset_vec.size(); i++){
            valid_dts += ","+dataset_vec[i];
        }
        std::cerr<<"terminated due to invalid dataset, valid datasets are: "<<valid_dts<<std::endl;
    }
    return is_valid_dataset;
}

void set_stds_means(Network_t* net){
    if(Configuration_deeppoly::dataset == "MNIST"){
        net->means.push_back(0);
        net->stds.push_back(1);
    }
    else if(Configuration_deeppoly::dataset == "CIFAR10"){
        net->means.push_back(0.4914);
        net->means.push_back(0.4822);
        net->means.push_back(0.4465);

        net->stds.push_back(0.2023);
        net->stds.push_back(0.1994);
        net->stds.push_back(0.2010);
    }
    else if(Configuration_deeppoly::dataset == "ACASXU"){
        net->means.push_back(19791.091);
        net->means.push_back(0.0);
        net->means.push_back(0.0);
        net->means.push_back(650.0);
        net->means.push_back(600.0);

        net->stds.push_back(60261.0);
        net->stds.push_back(6.28318530718);
        net->stds.push_back(6.28318530718);
        net->stds.push_back(1100.0);
        net->stds.push_back(1200.0);

    }
    else{
        assert(0 && "Invalid dataset in normalize image");
    }
}

void normalize_input_image(Network_t* net){
    xt::xarray<double> im = net->input_layer->res;
    std::string dataset = Configuration_deeppoly::dataset;
    std::vector<double> means = net->means;
    std::vector<double> stds = net->stds;
    if(dataset == "MNIST"){
        net->input_layer->res = (im - net->means[0])/net->stds[0];
    }
    else if(dataset == "CIFAR10"){
        size_t count = 0;
        xt::xarray<double> temp = xt::zeros<double>({Configuration_deeppoly::input_dim});
        for(size_t i=0; i<1024; i++){
            temp[count] = (im[count] - means[0])/stds[0];
            count += 1;
            temp[count] = (im[count] - means[1])/stds[1];
            count += 1;
            temp[count] = (im[count] - means[2])/stds[2];
            count += 1;
        }

        count = 0;
        for(size_t i=0; i<1024; i++){
            net->input_layer->res[i] = temp[count];
            count += 1;
            net->input_layer->res[i+1024] = temp[count];
            count += 1;
            net->input_layer->res[i+2048] = temp[count];
            count += 1;
        }
    }
    else if(dataset == "ACASXU"){
        for(size_t i=0; i<means.size(); i++){
            if(stds[i] != 0){
                net->input_layer->res[i] = (im[i] - means[i])/stds[i];
            }
        }
    }
    else{
        assert(0 && "Invalid dataset in normalization of image");
    }
}

void normalize_image(Network_t* net){
    xt::xarray<double> im = net->input_layer->res;
    std::string dataset = Configuration_deeppoly::dataset;
    std::vector<double> means = net->means;
    std::vector<double> stds = net->stds;
    if(dataset == "MNIST"){
        net->input_layer->res = (im - net->means[0])/net->stds[0];
    }
    else if(dataset == "CIFAR10"){
        for(size_t i=0; i<1024; i++){
            net->input_layer->res[i] = (im[i] - means[0])/stds[0];
            net->input_layer->res[i+1024] = (im[i+1024] - means[1])/stds[1];
            net->input_layer->res[i+2048] = (im[i+2048] - means[2])/stds[2];
        }
    }
    else if(dataset == "ACASXU"){
        for(size_t i=0; i<means.size(); i++){
            if(stds[i] != 0){
                net->input_layer->res[i] = (im[i] - means[i])/stds[i];
            }
        }
    }
    else{
        assert(0 && "Invalid dataset in normalization of image");
    }
}

void denormalize_image(Network_t* net){
    xt::xarray<double> im = net->input_layer->res;
    std::string dataset = Configuration_deeppoly::dataset;
    std::vector<double> means = net->means;
    std::vector<double> stds = net->stds;
    if(dataset == "MNIST"){
        net->input_layer->res = (im*net->stds[0]) + net->means[0];
    }
    else if(dataset == "CIFAR10"){
        for(size_t i=0; i<1024; i++){
            net->input_layer->res[i] = (im[i]*stds[0]) + means[0];
            net->input_layer->res[i+1024] = (im[i+1024]*stds[1]) + means[1];
            net->input_layer->res[i+2048] = (im[i+2048]*stds[2]) + means[2];
        } 
    }
    else if(dataset == "ACASXU"){
        for(size_t i=0; i<means.size(); i++){
            net->input_layer->res[i] = (im[i]*stds[i]) + means[i];
        }
    }
    else{
        assert(0 && "Invalid dataset in denormalization of image");
    }
}

int run_drefine_vnnlib(Network_t* net){
    VnnLib_t* vnn_lib = net->vnn_lib;
    int status = Verified;
    size_t loop_counter=0;
    bool is_verified_by_deeppoly = false;
    bool is_verified_by_drefine = false;
    auto start_time =  std::chrono::high_resolution_clock::now();
    for(Basic_pre_cond_t* pre_cond : vnn_lib->pre_cond_vec){
        is_verified_by_deeppoly = false;
        vnn_lib->out_prp->verified_sub_prp.clear();
        create_input_property_vnnlib(net, pre_cond);
        bool is_verified = run_deeppoly(net);
        if(!is_verified){
            std::tuple<int, size_t> ret_val = run_milp_refine_with_milp_mark_vnnlib(net);
            int ret = std::get<0>(ret_val);
            loop_counter = std::get<1>(ret_val);
            if(ret == Failed || ret == Unknown){
                status = ret;
                break;
            }
            else{
                is_verified_by_drefine = true;
            }
        }
        else if(!is_verified_by_drefine){
            is_verified_by_deeppoly = true;
        }
    }
    if(status == Failed){
        print_status_string(net, 0, "drefine", 0, loop_counter, start_time);
    }
    else if(status == Unknown){
        print_status_string(net, 2, "drefine", 0, loop_counter, start_time);
    }
    else{
        if(is_verified_by_deeppoly){
            print_status_string(net, 1, "deeppoly", 0, loop_counter, start_time);
        }
        else{
            print_status_string(net, 1, "drefine", 0, loop_counter, start_time);
        }
        status = Verified;
    }
    return status;
}

std::tuple<int, size_t> run_milp_refine_with_milp_mark_vnnlib(Network_t* net){
    int status;
    size_t loop_upper_bound = MILP_WITH_MILP_LIMIT;
    size_t loop_counter = 1;
    bool is_bound_exceeded = true;
    while(loop_counter < loop_upper_bound){
        bool is_ce = run_milp_mark_with_milp_refine(net);
        std::cout<<"refinement iteration: "<<loop_counter<<std::endl;
        if(is_ce){
            is_bound_exceeded = false;
            status = Failed;
            break;
        }
        else{
            bool is_verified = is_prp_verified_by_milp(net);
            if(is_verified){
                is_bound_exceeded = false;
                status = Verified;
                break;
            }
        }
        loop_counter++;
    }
    if(is_bound_exceeded){
        status = Unknown;
    }
    std::tuple<int, size_t> tup1(status, loop_counter);
    return tup1;
}

double get_random_val(double low, double high){
    double r = low + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(high-low)));
    return r;
}

xt::xarray<double> get_random_images(Network_t* net){
    std::vector<double> vals;
    double v;
    vals.reserve((net->input_dim)*NUM_RANDOM_IMAGES);
    for(size_t i=0; i<NUM_RANDOM_IMAGES; i++){
        for(size_t j=0; j<net->input_dim; j++){
            Neuron_t* nt = net->input_layer->neurons[j];
            v = get_random_val(-nt->lb, nt->ub);
            vals.push_back(v);
        }
    }
    std::vector<size_t> shape = {NUM_RANDOM_IMAGES, net->input_dim};
    xt::xarray<double> temp = xt::adapt(vals, shape);
    // temp = xt::transpose(temp);
    // std::cout<< xt::adapt(temp.shape()) << std::endl;
    return temp;
}

bool is_ce_cheap_check(Network_t* net){
    xt::xarray<double> images = get_random_images(net);
    net->forward_propgate_network(0, images);
    xt::xarray<double> res = net->layer_vec.back()->res;
    size_t count = 0;
    std::vector<double> row_double;
    row_double.resize(net->output_dim);

    for(size_t i=0; i<(NUM_RANDOM_IMAGES*net->output_dim); i++){
        if(count < net->output_dim){
            row_double[count] = res[i];
            count++;
        }
        if(count == net->output_dim){
            size_t max_ind = 0;
            double max_elem = row_double[0];
            // std::cout<<row_double[0]<<" ";
            for(size_t j=1; j<net->output_dim; j++){
                if(max_elem < row_double[j]){
                    max_ind = j;
                    max_elem = row_double[j];
                }
                // std::cout<<row_double[j]<<" ";
            }
            if(max_ind != net->actual_label){
                return true;
            }
            count = 0;
        }
    }
    return false;
}

