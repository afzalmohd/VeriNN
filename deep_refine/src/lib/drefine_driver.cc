#include "../../deeppoly/helper.hh"
#include "../../deeppoly/vnnlib.hh"
#include "../../deeppoly/parser.hh"
#include "drefine_driver.hh"
#include "milp_refine.hh"
#include "milp_mark.hh"
#include "vericomp.hh"
#include<cmath>
#include<fstream>
#include<iostream>
#include<chrono>
#include<queue>
#include "../../k/findk.hh"
#include "concurrent_run.hh"
#include "bounds_milp.hh"


int run_refine_poly(int num_args, char* params[]){
    int is_help = deeppoly_set_params(num_args, params);
    if(is_help || (!is_valid_dataset())){
        return 1;
    }

    // find_k();
    // return 0;

    Configuration_deeppoly::is_target_ce = Configuration_deeppoly::is_target_ce && Configuration_deeppoly::is_conf_ce;

    Network_t* net;
    std::chrono::_V2::system_clock::time_point start_time;
    drefine_status status;
    
    if(Configuration_deeppoly::vnnlib_prp_file_path != "" && Configuration_deeppoly::bounds_path != ""){
        net = deeppoly_initialize_network();
        parse_vnnlib_simplified_mnist(net, Configuration_deeppoly::vnnlib_prp_file_path);
        update_last_layer(net);
        bounds_parser(net, Configuration_deeppoly::bounds_path);
        start_time =  std::chrono::high_resolution_clock::now();
        status = run_milp_refine_with_milp_mark_ab(net);
        // return status;
    }
    else if(Configuration_deeppoly::vnnlib_prp_file_path != ""){
        VnnLib_t* verinn_lib = parse_vnnlib(Configuration_deeppoly::vnnlib_prp_file_path);
        net = deeppoly_initialize_network();
        net->vnn_lib = verinn_lib;
        start_time =  std::chrono::high_resolution_clock::now();
        status = run_drefine_vnnlib(net);
    }
    else{
        net = deeppoly_initialize_network();
        set_stds_means(net);
        start_time =  std::chrono::high_resolution_clock::now();
        bool is_same_label = is_actual_and_pred_label_same(net, Configuration_deeppoly::image_index);
        if(!is_same_label){
            return 0;
        }

        // return 0;
        if(Configuration_deeppoly::is_target_ce){
            if(net->pred_label == TARGET_CLASS){
                return 0;
            }
        }
        // std::cout<<"Check..."<<std::endl;
        // compute_pre_imp(net);
        // return 0;
        create_input_prop(net);
        std::queue<Network_t*> work_q;
        work_q.push(net);
        status = run_refine_poly1(work_q);
    }
      
    size_t image_index = Configuration_deeppoly::image_index;
    std::string tool_name = Configuration_deeppoly::tool;
    if(status == DEEPPOLY_VERIFIED){
        tool_name = "deeppoly";
    }
    print_status_string(net, status, tool_name, image_index, 0, start_time);

    return 0;
}

void set_confidence(Network_t* net){
    double orig_im_conf = 0;
    if(Configuration_deeppoly::is_softmax_conf_ce){
        orig_im_conf = compute_softmax_conf(net, net->actual_label);
    }
    else{
        orig_im_conf = compute_conf(net, net->actual_label);
    }
    Global_vars::orig_im_conf = orig_im_conf;
    std::cout<<"Correct image conf: "<<orig_im_conf<<std::endl;

    if(Configuration_deeppoly::conf_value == -1.0){
        Configuration_deeppoly::conf_value = orig_im_conf;
        Configuration_deeppoly::softmax_conf_value = orig_im_conf;
    }

    if(Configuration_deeppoly::conf_value == 0){
        Configuration_deeppoly::is_conf_ce = false;
        Configuration_deeppoly::is_softmax_conf_ce = false;
    }
}

bool is_actual_and_pred_label_same(Network_t* net, size_t image_index){
    std::cout<<"Image index: "<<image_index<<std::endl;
    std::string image_str = get_image_str(Configuration_deeppoly::dataset_path, image_index);
    deeppoly_parse_input_image_string(net, image_str);
    // normalize_image(net);
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
    set_confidence(net);
    return true;
}

drefine_status run_refine_poly1(std::queue<Network_t*>& work_q){
    bool verified_by_deeppoly = false;
    size_t counter = 0;
    while(!work_q.empty()){
        counter++;
        std::cout<<"Queue iteration: "<<counter<<std::endl;
        std::cout<<"work_q size: "<<work_q.size()<<std::endl;
        Network_t* net = work_q.front();
        work_q.pop();
        drefine_status status = run_refine_poly_for_one_task(net);
        size_t num_marked_nt = num_marked_neurons(net);
        if(!Configuration_deeppoly::is_reset_marked_nts){
            Global_vars::num_marked_neurons += num_marked_nt;
        }
        Global_vars::sub_prob_counts += 1;

        if(status == DEEPPOLY_VERIFIED && Global_vars::sub_prob_counts == 1){
            verified_by_deeppoly = true;
        }
        if(status == VERIFIED || status == DEEPPOLY_VERIFIED){
            continue;
        }
        else if(status == FAILED){
            return FAILED;
        }
        else{ // status is unknown
            if(Configuration_deeppoly::is_input_split){
                create_problem_instances(net, work_q);
            }
            else{
                return UNKNOWN;
            }
            
        }
    }
    if(verified_by_deeppoly){
        return DEEPPOLY_VERIFIED;
    }
    else{
        return VERIFIED;
    }
}

bool is_val_exist_in_vec_size_t(size_t val, std::vector<size_t>& vec){
    for(size_t v : vec){
        if(v == val){
            return true;
        }
    }
    return false;
}

void set_dims_to_split(Network_t* net){
    /* 
    //Taking output dim which has max error
    size_t index_with_max_err;
    double max_val = -INFINITY;
    for(auto itr = net->index_vs_err.begin(); itr != net->index_vs_err.end(); itr++){
        if(max_val < itr->second){
            index_with_max_err = itr->first;
            max_val = itr->second;
        }
    }
    net->dims_to_split = net->index_map_dims_to_split[index_with_max_err];
    */
    net->dims_to_split = net->index_map_dims_to_split[net->counter_class_dim]; //Take dims where fails only
    if(Configuration_deeppoly::vnnlib_prp_file_path != ""){
        Layer_t* input_layer = net->input_layer;
        std::vector<size_t> index_vec;
        for(size_t i=0; i<MAX_INPUT_DIMS_TO_SPLIT; i++){
            double max_val = -INFINITY;
            size_t max_val_index;
            for(size_t j=0; j<input_layer->dims; j++){
                Neuron_t* nt = input_layer->neurons[j];
                double lb = -nt->lb;
                double delta = nt->ub - lb;
                if(max_val <= delta && !is_val_exist_in_vec_size_t(j, index_vec)){
                    max_val_index = j;
                    max_val = delta;
                }
            }
            index_vec.push_back(max_val_index);
        }
        net->dims_to_split = index_vec;
    }
}

void create_problem_instances(Network_t* net, std::queue<Network_t*>& work_q){
    set_dims_to_split(net);
    size_t num_dims_to_split = net->dims_to_split.size();
    if(num_dims_to_split <= 0){
        return;
    }
    // std::cout<<"Dims to split: ";
    // for(size_t val : net->dims_to_split){
    //     std::cout<<val<<" ";
    // }
    // std::cout<<std::endl;
    size_t a[num_dims_to_split];

    std::cout<<"Parantes net splitted dims: ";
    for(size_t i=0; i<net->input_layer->dims; i++){
        if(is_dim_to_split(i, net->dims_to_split)){
            Neuron_t* nt = net->input_layer->neurons[i];
            std::cout<<i<<": ("<<-nt->lb<<","<<nt->ub<<") ";
        }
    }
    std::cout<<std::endl;
    create_problem_instances_recursive(net, work_q, num_dims_to_split, a, 0);
}

void create_problem_instances_recursive(Network_t* net, std::queue<Network_t*>& work_q, size_t n, size_t a[], size_t i){
    if(n == i){
        create_one_problem_instances_input_split(net, work_q, n, a);
        return;
    }
    a[i] = 0;
    create_problem_instances_recursive(net, work_q, n, a, i+1);

    a[i] = 1;
    create_problem_instances_recursive(net, work_q, n, a, i+1);
}

void create_one_problem_instances_input_split(Network_t* net, std::queue<Network_t*>& work_q, size_t n, size_t a[]){
    Network_t* net1 = new Network_t();
    copy_network(net1, net);
    net1->verified_out_dims = net->verified_out_dims;
    Layer_t* input_layer = net1->input_layer;
    size_t j=0;
    for(size_t i=0; i<input_layer->dims; i++){
        Neuron_t* nt = input_layer->neurons[i];
        Neuron_t* old_nt = net->input_layer->neurons[i];
        nt->lb = old_nt->lb;
        nt->ub = old_nt->ub;
        if(is_dim_to_split(i, net->dims_to_split)){
            double lb = -nt->lb;
            double ub = nt->ub;
            double mid = (lb+ub)/2;
            if(a[j] == 0){
                nt->ub = mid;
            }
            else{
                nt->lb = -mid;
            }
            j++;
        }
    }
    std::cout<<"Splitted dims: ";
    for(size_t i=0; i<net1->input_layer->dims; i++){
        if(is_dim_to_split(i, net->dims_to_split)){
            Neuron_t* nt = net1->input_layer->neurons[i];
            std::cout<<i<<": ("<<-nt->lb<<","<<nt->ub<<") ";
        }
    }
    std::cout<<std::endl;
    work_q.push(net1);
}

bool is_dim_to_split(size_t i, std::vector<size_t> dims){
    for(size_t val : dims){
        if(i == val){
            return true;
        }
    }
    return false;
}

drefine_status run_refine_poly_for_one_task(Network_t* net){
    std::string bounds_file_path = "";
    // parse_file_and_update_bounds(net, bounds_file_path);
    bool is_verified = run_deeppoly(net);
    // net->print_network();
    if(is_verified){
        return DEEPPOLY_VERIFIED;
    }

    if(Configuration_deeppoly::tool == "deeppoly"){
        return UNKNOWN;
    }
    
    if(IS_BOUND_TIGHTENING_MILP){
        forward_analysis_bounds_milp_seq(net);
        // net->print_network();
        // is_verified = run_deeppoly(net);
        // net->print_network();
        // if(is_verified){
        //     return VERIFIED;
        // }
    }

    drefine_status status;
    if(Configuration_deeppoly::vnnlib_prp_file_path != ""){
        status = run_milp_refine_with_milp_mark_vnnlib(net);
    }
    else{
        status = run_cegar_milp_mark_milp_refine(net);
        // status = run_milp_refine_with_milp_mark_input_split(net);
    }
    return status;
}

void print_logs(Network_t* net){
    for(Layer_t* layer : net->layer_vec){
        if(layer->is_marked){
            std::cout<<"Layer index::: "<<layer->layer_index<<", ";
            for(Neuron_t* nt : layer->neurons){
                if(nt->is_marked){
                    std::cout<<nt->neuron_index<<" ";
                }
            }
            std::cout<<std::endl;
        }
    }
    std::cout<<std::endl;

    std::cout<<"Iteration number: "<<Global_vars::iter_counts<<std::endl;
    for(Layer_t* layer : net->layer_vec){
        if(layer->is_marked){
            std::cout<<"Layer index::: "<<layer->layer_index<<",";
            size_t count = 0;
            for(Neuron_t* nt : layer->neurons){
                if(nt->is_marked){
                    count++;
                }
            }
            std::cout<<count<<std::endl;
        }
    }
    std::cout<<std::endl;
}

drefine_status run_cegar_milp_mark_milp_refine(Network_t* net){
    size_t loop_upper_bound = MILP_WITH_MILP_LIMIT;
    drefine_status status = UNKNOWN;
    // status = is_verified_by_vericomp(net);
    if(status != UNKNOWN){
        return status;
    }

    if(Configuration_deeppoly::is_concurrent){
        Configuration_deeppoly::is_reset_marked_nts = false;
        bool is_image_verified = is_image_verified_by_milp(net);
        if(is_image_verified){
            return VERIFIED;
        }
        // bool is_ce = run_milp_mark_with_milp_refine(net);
        bool is_ce = run_refinement_cegar(net);
        if(is_ce){
            return FAILED;
        }
        
        std::vector<int > prev_comb;
        if(rec_con(net,prev_comb,Global_vars::new_marked_nts.size())){
            return VERIFIED;
        }
        else{
            return FAILED;
        }
        return UNKNOWN;
    }
    
    size_t loop_counter = 0;
    while(loop_counter < loop_upper_bound){
        auto start_time = std::chrono::high_resolution_clock::now();
        bool is_image_verified = is_image_verified_by_milp(net);
        auto end_time = std::chrono::high_resolution_clock::now();
        Global_vars::refinement_time += std::chrono::duration<double>(end_time - start_time);
        if(is_image_verified){
            return VERIFIED;
        }
        start_time = std::chrono::high_resolution_clock::now();
        bool is_ce = run_refinement_cegar(net);
        end_time = std::chrono::high_resolution_clock::now();
        Global_vars::marking_time += std::chrono::duration<double>(end_time - start_time);
        Global_vars::iter_counts += 1;
        IFVERBOSE(print_logs(net));
        
        std::cout<<"MARK_NEURONS_TIME: "<<std::to_string(Global_vars::marking_time.count())<<" , REFINEMENT_TIME: "<<std::to_string(Global_vars::refinement_time.count())<<std::endl;
        if(is_ce){
            return FAILED;
        }
        loop_counter += 1;
    }

    return status;
}


drefine_status run_milp_refine_with_milp_mark_input_split(Network_t* net){
    size_t loop_upper_bound = MILP_WITH_MILP_LIMIT;
    if(Configuration_deeppoly::is_input_split && Global_vars::sub_prob_counts < 3){
        loop_upper_bound = MILP_WITH_MILP_LIMIT_WITH_INPUT_SPLIT;
    }
    bool generate_data = false;
    drefine_status status = UNKNOWN;
    // status = is_verified_by_vericomp(net);
    if(status != UNKNOWN){
        return status;
    }

    // return status;
    xt::xarray<double> prev_input_point = net->input_layer->res;
    size_t loop_counter = 0;
    while(loop_counter < loop_upper_bound){
        bool is_ce;
        auto start_time = std::chrono::high_resolution_clock::now();
        if(IS_LIGHT_WEIGHT_MARKED_ANALYSIS){
            is_ce = mark_neurons_with_light_analysis(net);
        }
        else{
            is_ce = run_milp_mark_with_milp_refine(net);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        Global_vars::marking_time += std::chrono::duration<double>(end_time - start_time);
        if(is_ce){
            if(generate_data){
                print_image_with_label(net, prev_input_point);
            }
            return FAILED;
        }
        else{
            Global_vars::iter_counts += 1;
            print_logs(net);
            get_images_from_satval(prev_input_point, net->input_layer);
            if(loop_counter > 0 && generate_data){
                 print_image_with_label(net, prev_input_point);
            }
            start_time = std::chrono::high_resolution_clock::now();
            bool is_image_verified = is_image_verified_by_milp(net);
            end_time = std::chrono::high_resolution_clock::now();
            Global_vars::refinement_time += std::chrono::duration<double>(end_time - start_time);
            std::cout<<"MARK_NEURONS_TIME: "<<std::to_string(Global_vars::marking_time.count())<<" , REFINEMENT_TIME: "<<std::to_string(Global_vars::refinement_time.count())<<std::endl;
            if(is_image_verified){
               return VERIFIED;
            }
        }
        loop_counter++;
    }
    return UNKNOWN;
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
    std::cout<<file_path<<std::endl;
    std::ofstream my_file;
    my_file.open(file_path, std::ios::app);
    if(my_file.is_open()){
        my_file<<s<<std::endl;
        my_file.close();
    }
    else{
        std::cout<<"result file could not open"<<std::endl;
        assert(0 && "result file could not open\n");
    }
}

void print_status_string(Network_t* net, size_t tool_status, std::string tool_name, size_t image_index, size_t loop_counter, std::chrono::_V2::system_clock::time_point start_time){
    auto end_time = std::chrono::high_resolution_clock::now();
    std::string status_string;
    if(tool_status == FAILED){
        status_string = "failed";
    }
    else if(tool_status == VERIFIED || tool_status == DEEPPOLY_VERIFIED){
        status_string = "verified";
    }
    else{
        status_string = "unknown";
    }
    std::chrono::duration<double> duration = std::chrono::duration<double>(end_time - start_time);
    std::string base_net_name = get_absolute_file_name_from_path(Configuration_deeppoly::net_path);
    std::string base_prp_name = get_absolute_file_name_from_path(Configuration_deeppoly::vnnlib_prp_file_path);
    // size_t num_marked_nt = num_marked_neurons(net);
    if(base_prp_name == ""){
        base_prp_name = "null";
    }
    std::string str = base_net_name+","+std::to_string(Configuration_deeppoly::epsilon)+","+std::to_string(image_index)+","+std::to_string(net->pred_label)+","+base_prp_name+","+status_string+","+tool_name+","+std::to_string(Global_vars::sub_prob_counts)+","+std::to_string(Global_vars::iter_counts)+","+std::to_string(Global_vars::num_marked_neurons)+","+std::to_string(Configuration_deeppoly::conf_value)+","+std::to_string(Global_vars::orig_im_conf)+","+std::to_string(Global_vars::ce_im_conf)+","+std::to_string(duration.count())+","+std::to_string(Global_vars::marking_time.count())+","+std::to_string(Global_vars::refinement_time.count());
    write_to_file(Configuration_deeppoly::result_file, str);
    str = base_net_name+","+std::to_string(Configuration_deeppoly::epsilon)+",image_index="+std::to_string(image_index)+",image_label="+std::to_string(net->pred_label)+",prop_name="+base_prp_name+","+status_string+","+tool_name+",num_sub_prob="+std::to_string(Global_vars::sub_prob_counts)+",num_cegar_iterations:"+std::to_string(Global_vars::iter_counts)+",num_marked_neurons="+std::to_string(Global_vars::num_marked_neurons)+",given conf="+std::to_string(Configuration_deeppoly::conf_value)+",orig image conf="+std::to_string(Global_vars::orig_im_conf)+",ce image conf="+std::to_string(Global_vars::ce_im_conf)+",total_time="+std::to_string(duration.count())+",marking_time="+std::to_string(Global_vars::marking_time.count())+",refinement_time="+std::to_string(Global_vars::refinement_time.count());
    std::cout<<str<<std::endl;
}

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

        // count = 0;
        // for(size_t i=0; i<3072; i++){
        //     if(count < 1024){
        //         net->input_layer->res[i] = (im[i] - means[0])/stds[0];
        //     }
        //     else if(count >= 1024 && count < 2048){
        //         net->input_layer->res[i] = (im[i] - means[1])/stds[1];
        //     }
        //     else{
        //         net->input_layer->res[i] = (im[i] - means[2])/stds[2];
        //     }
        // }
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

// drefine_status run_milp_refine_with_milp_mark_vnnlib_input_split(Network_t* net){
//     std::queue<Network_t*> work_q;
//     work_q.push(net);
//     drefine_status status = run_refine_poly1(work_q);
//     return status;
// }

drefine_status run_drefine_vnnlib(Network_t* net){
    VnnLib_t* vnn_lib = net->vnn_lib;
    drefine_status status = VERIFIED;
    // bool is_verified_by_deeppoly = false;
    bool is_verified_by_drefine = false;
    for(Basic_pre_cond_t* pre_cond : vnn_lib->pre_cond_vec){
        vnn_lib->out_prp->verified_sub_prp.clear();
        create_input_property_vnnlib(net, pre_cond);
        std::queue<Network_t*> work_q;
        work_q.push(net);
        status = run_refine_poly1(work_q);
        if(status == FAILED || status == UNKNOWN){
            break;
        }
        else if(status == VERIFIED){
            is_verified_by_drefine = true;
        }
    }
    if(status == VERIFIED || status == DEEPPOLY_VERIFIED){
        if(is_verified_by_drefine){
            return VERIFIED;
        }
        else{
            return DEEPPOLY_VERIFIED;
        }
    }
    return status;
}

drefine_status run_milp_refine_with_milp_mark_vnnlib(Network_t* net){
    drefine_status status;

    size_t loop_upper_bound = MILP_WITH_MILP_LIMIT;
    if(Configuration_deeppoly::is_input_split){
        loop_upper_bound = MILP_WITH_MILP_LIMIT_WITH_INPUT_SPLIT;
    }

    size_t loop_counter = 1;
    bool is_bound_exceeded = true;
    while(loop_counter < loop_upper_bound){
        auto start_time = std::chrono::high_resolution_clock::now();
        bool is_ce = run_milp_mark_with_milp_refine(net);
        auto end_time = std::chrono::high_resolution_clock::now();
        Global_vars::marking_time += std::chrono::duration<double>(end_time - start_time);
        std::cout<<"refinement iteration: "<<loop_counter<<std::endl;
        if(is_ce){
            is_bound_exceeded = false;
            status = FAILED;
            break;
        }
        else{
            start_time = std::chrono::high_resolution_clock::now();
            bool is_verified = is_prp_verified_by_milp(net);
            end_time = std::chrono::high_resolution_clock::now();
            Global_vars::refinement_time += std::chrono::duration<double>(end_time - start_time);
            if(is_verified){
                is_bound_exceeded = false;
                status = VERIFIED;
                break;
            }
        }
        loop_counter++;
        Global_vars::iter_counts += 1;
    }
    if(is_bound_exceeded){
        status = UNKNOWN;
    }
    // std::tuple<int, size_t> tup1(status, loop_counter);
    return status;
}

drefine_status run_milp_refine_with_milp_mark_ab(Network_t* net){
    drefine_status status;

    size_t loop_upper_bound = MILP_WITH_MILP_LIMIT;
    if(Configuration_deeppoly::is_input_split){
        loop_upper_bound = MILP_WITH_MILP_LIMIT_WITH_INPUT_SPLIT;
    }

    size_t loop_counter = 1;
    bool is_bound_exceeded = true;
    while(loop_counter < loop_upper_bound){
        auto start_time = std::chrono::high_resolution_clock::now();
        bool is_verified = is_prp_verified_ab(net);
        auto end_time = std::chrono::high_resolution_clock::now();
        Global_vars::refinement_time += std::chrono::duration<double>(end_time - start_time);
        if(is_verified){
            is_bound_exceeded = false;
            status = VERIFIED;
            break;
        }
        start_time = std::chrono::high_resolution_clock::now();
        bool is_ce = run_milp_mark_with_milp_refine(net);
        end_time = std::chrono::high_resolution_clock::now();
        Global_vars::marking_time += std::chrono::duration<double>(end_time - start_time);
        std::cout<<"refinement iteration: "<<loop_counter<<std::endl;
        Global_vars::iter_counts += 1;

        if(is_ce){
            is_bound_exceeded = false;
            status = FAILED;
            break;
        }
        loop_counter++;
    }
    if(is_bound_exceeded){
        status = UNKNOWN;
    }
    // std::tuple<int, size_t> tup1(status, loop_counter);
    Global_vars::num_marked_neurons = num_marked_neurons(net);
    return status;
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

bool is_exists_in_vector(size_t val, std::vector<size_t>& vec){
    for(size_t v : vec){
        if(v == val){
            return true;
        }
    }
    return false;
}

xt::xarray<double> delete_rows_weight(xt::xarray<double>& mat, std::vector<size_t>& del_rows, size_t num_rows, size_t num_cols){
    size_t effective_num_rows = num_rows - del_rows.size();
    assert(effective_num_rows >= 0 && "Row index out of range\n");
    std::vector<double> vals;
    vals.reserve(effective_num_rows*num_cols);
    size_t row_count=0;
    size_t col_count=0;
    bool is_row_to_be_deleted = false;
    for(size_t i=0; i<num_rows*num_cols; i++){
        is_row_to_be_deleted = is_exists_in_vector(row_count, del_rows);
        if(!is_row_to_be_deleted){
            double v = mat[i];
            vals.push_back(v);
        }
        col_count++;
        if(col_count == num_cols){
            row_count++;
            col_count=0;
        }
    }
    std::vector<size_t> shape = {effective_num_rows, num_cols};
    xt::xarray<double> res_mat = xt::adapt(vals, shape);
    return res_mat;
}

xt::xarray<double> delete_rows_bias(xt::xarray<double>& mat, std::vector<size_t>& del_rows, size_t size){
    size_t effective_size = size - del_rows.size();
    assert(effective_size >= 0 && "Row index out of range\n");
    std::vector<double> vals;
    vals.reserve(effective_size);
    for(size_t i=0; i<size; i++){
        bool is_index_to_be_deleted = is_exists_in_vector(i, del_rows);
        if(!is_index_to_be_deleted){
            double v = mat[i];
            vals.push_back(v);
        }
    }
    std::vector<size_t> shape = {effective_size};
    xt::xarray<double> res_mat = xt::adapt(vals, shape);
    return res_mat;
}

std::map<size_t, std::vector<size_t>> get_non_essential_neurons(Network_t* net){
    std::map<size_t, std::vector<size_t>> mark;
    for(size_t i=0; i<net->layer_vec.size()-2; i++){
        Layer_t* layer = net->layer_vec[i];
        std::vector<size_t> vec1;
        std::vector<size_t> vec2;
        if(!layer->is_activation){
            for(size_t j=0; j<layer->neurons.size(); j++){
                Neuron_t* nt = layer->neurons[j];
                if(nt->ub <= 0){
                    vec1.push_back(j);
                    vec2.push_back(j);
                }
            }
        }
        if(vec1.size() > 0){
            mark[i] = vec1;
            mark[i+1] = vec2;
        }
    }
    return mark;
}

void init_neurons(Network_t* net){
    for(Layer_t* layer : net->layer_vec){
        for(size_t i=0; i<layer->neurons.size(); i++){
            Neuron_t* nt = layer->neurons[i];
            delete nt;
        }
        layer->neurons.clear();
        for(size_t i=0; i<layer->dims; i++){
            Neuron_t* nt = new Neuron_t();
            nt->neuron_index = i;
            nt->layer_index = layer->layer_index;
            layer->neurons.push_back(nt);
        }
    }
}

void update_weight_matrices(Network_t* net, std::map<size_t, std::vector<size_t>>& mark_map){
    for(auto i : mark_map){
        size_t layer_index = i.first;
        std::vector<size_t> del_indexes = i.second;
        Layer_t* layer = net->layer_vec[layer_index];
        if(!layer->is_activation){
            Layer_t* next_affine_layer = net->layer_vec[layer_index+2];
            xt::xarray<double> curr_weight = layer->w;
            xt::xarray<double> curr_bias = layer->b;
            curr_weight = xt::transpose(curr_weight);
            curr_weight = delete_rows_weight(curr_weight, del_indexes, layer->dims, layer->pred_layer->dims);
            layer->w = xt::transpose(curr_weight);
            curr_bias = delete_rows_bias(curr_bias, del_indexes, layer->dims);
            layer->b = curr_bias;

            xt::xarray<double> next_layer_w = next_affine_layer->w;
            next_layer_w = delete_rows_weight(next_layer_w, del_indexes, layer->dims, next_affine_layer->dims);
            next_affine_layer->w = next_layer_w;

            layer->dims = layer->dims - del_indexes.size();
            net->layer_vec[layer_index+1]->dims = layer->dims;
            layer->w_shape = {layer->pred_layer->dims, layer->dims};
            next_affine_layer->w_shape = {layer->dims, next_affine_layer->dims};
        }
    } 
}


void remove_non_essential_neurons(Network_t* net){
    std::map<size_t, std::vector<size_t>> mark_mp = get_non_essential_neurons(net);
    for(auto i : mark_mp){
        std::cout<<"Layer index: "<<i.first<<", removed neurons: "<<i.second.size()<<std::endl;
    }
    update_weight_matrices(net, mark_mp);
    init_neurons(net);
    for(Layer_t* layer : net->layer_vec){
        if(!layer->is_activation){
            std::cout<<"Layer index: "<<layer->layer_index<<", weight: "<<xt::adapt(layer->w.shape())<<" , bias: "<<xt::adapt(layer->b.shape())<<std::endl;
        }
    }
}

void create_input_prop(Network_t* net){
    Layer_t* layer = net->input_layer;
    double ep = Configuration_deeppoly::epsilon;
    for(size_t i=0; i < layer->dims; i++){
        Neuron_t* nt = layer->neurons[i];
        nt->ub = layer->res[i] + ep;
        nt->lb = layer->res[i] - ep;
        if(nt->ub > 1.0){
            nt->ub = 1.0;
        }
        if(nt->lb < 0.0){
            nt->lb = 0.0;
        }
        nt->lb = -nt->lb;
    }
    if(Configuration_deeppoly::is_small_ex){
        for(size_t i=0; i< layer->dims; i++){
            Neuron_t* nt = layer->neurons[i];
            nt->ub = 1;
            nt->lb = 1;//actual is -1
        }
    }
}

void copy_network(Network_t* net1, Network_t* net){
    net1->numlayers = net->numlayers;
    net1->input_dim = net->input_dim;
    net1->output_dim = net->output_dim;
    net1->actual_label = net->actual_label;
    net1->pred_label = net->pred_label;
    net1->stds = net->stds;
    net1->means = net->means;
    net1->vnn_lib = net->vnn_lib;
    net1->input_layer = new Layer_t();
    copy_layer(net1->input_layer, net->input_layer);
    for(size_t i=0; i<net1->numlayers; i++){
        Layer_t* layer1 = new Layer_t();
        copy_layer(layer1, net->layer_vec[i]);
        net1->layer_vec.push_back(layer1);
        if(i==0){
            layer1->pred_layer = net1->input_layer;
        }
        else{
            layer1->pred_layer = net1->layer_vec[i-1];
        }
    }
}

void copy_layer(Layer_t* layer1, Layer_t* layer){
    layer1->activation = layer->activation;
    layer1->dims = layer->dims;
    layer1->is_activation = layer->is_activation;
    layer1->layer_index = layer->layer_index;
    layer1->layer_type = layer->layer_type;
    if(layer1->layer_index == -1){ // input layer
        layer1->res = layer->res;
    }
    else{
        layer1->b = layer->b;
        layer1->w = layer->w;
        layer1->w_shape = layer->w_shape;
    }
    for(size_t i=0; i<layer1->dims; i++){
        Neuron_t* nt = new Neuron_t();
        nt->neuron_index = i;
        nt->layer_index = layer1->layer_index;
        layer1->neurons.push_back(nt);
    }
}

void print_image_with_label(Network_t* net, xt::xarray<double>& prev_input_point){
    std::string file_path = "/users/rs/afzal/tools/VeriNN/deep_refine/gen_data_ab_unsolved.csv";
    std::string file_path1 = "/users/rs/afzal/tools/VeriNN/deep_refine/seq_ab_unsolved.csv";
    double ep = 0;
    Layer_t* input_layer = net->input_layer;
    for(size_t i=0; i<net->input_dim; i++){
        double satval = input_layer->neurons[i]->sat_val;
        double diff = abs(satval - prev_input_point[i]);
        if(diff > ep){
            ep = diff;
        }
    }
    std::cout<<"Epsilon: "<<ep<<std::endl;

    net->forward_propgate_network(0, prev_input_point);
    for(size_t i=0; i<net->output_dim; i++){
        std::cout<<net->layer_vec.back()->res[i]<<" "<<std::endl;
    }

    std::string data = std::to_string(net->actual_label);
    for(size_t i=0; i<net->input_dim; i++){
        // size_t val = (size_t)prev_input_point[i];
        data += ","+std::to_string(prev_input_point[i]);
    }
    std::cout<<"Data: "<<std::endl;
    std::cout<<data<<std::endl;
    write_to_file(file_path, data);
    data = Configuration_deeppoly::net_path+","+std::to_string(Configuration_deeppoly::epsilon);
    write_to_file(file_path1, data);
}