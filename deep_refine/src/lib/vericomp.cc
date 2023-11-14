#include "vericomp.hh"
#include<xtensor/xio.hpp>
#include<fstream>
#include<iostream>
#include "milp_refine.hh"


std::vector<std::vector<double>> compute_pre_imp(Network_t* net){
    std::vector<xt::xarray<double>> prev_imp_vec;
    std::vector<double> vec(10, 1.0);
    std::vector<size_t> shape = {10,1};
    xt::xarray<double> prev_comp = xt::adapt(vec, shape);
    for(int i=net->layer_vec.size()-1; i>=0; i--){
        Layer_t* layer = net->layer_vec[i];
        if(!layer->is_activation){
            xt::xarray<double> imp = xt::linalg::dot(layer->w, prev_comp);
            prev_imp_vec.push_back(imp);
            prev_comp = imp;
        }
    }

    std::vector<std::vector<double>> imp_vec_vec;
    bool is_first = true;
    size_t layer_counter = 0;
    for(int i=net->layer_vec.size() -1; i>=0; i--){
        Layer_t* layer = net->layer_vec[i];
        if(!layer->is_activation){
            if(is_first){
                is_first = false;
                continue;
            }
            xt::xarray<double> prev_imp = prev_imp_vec[layer_counter];
            std::vector<double> imp_vec;
            auto imp_iter = prev_imp.begin();
            for(size_t j=0; j<layer->neurons.size(); j++, imp_iter++){
                Neuron_t* nt  = layer->neurons[j];
                double lb = -nt->lb;
                double range = nt->ub - lb;
                double imp1 = *imp_iter;
                imp_vec.push_back(range*imp1);
            }
            layer_counter++;
            imp_vec_vec.push_back(imp_vec);
        }
    }

    xt::xarray<double> prev_imp = prev_imp_vec[layer_counter];
    std::vector<double> imp_vec;
    auto imp_iter = prev_imp.begin();
    for(size_t i=0; i< net->input_dim; i++, imp_iter++){
        Neuron_t* nt = net->input_layer->neurons[i];
        double lb = -nt->lb;
        double range = nt->ub - lb;
        double imp1 = *imp_iter;
        imp_vec.push_back(imp1*range);
    }

    imp_vec_vec.push_back(imp_vec);
    return imp_vec_vec;

}

size_t get_max_idx(size_t idx, std::vector<double>& vec){
    size_t selected_idx = idx;
    double max_val = vec[idx];
    for(size_t i=idx+1; i<vec.size(); i++){
        if(max_val < vec[i]){
            max_val = vec[i];
            selected_idx = i;
        }
    }
    
    return selected_idx;
}

std::vector<size_t> imp_based_sorted_index(std::vector<double> vec){
    std::vector<size_t> index_vec;
    for(size_t i=0; i<vec.size(); i++){
        size_t max_idx= get_max_idx(i, vec);
        double val = vec[i];
        vec[i] = vec[max_idx];
        vec[max_idx] = val;
        index_vec.push_back(max_idx);
    }
    return index_vec;
}

void update_data(std::map<size_t, std::map<size_t, std::vector<size_t>>>& vericomp_data, size_t image_idx, size_t nt1, size_t nt2, size_t nt3){
    auto it = vericomp_data.find(image_idx);
    if(it == vericomp_data.end()){
        std::vector<size_t> vec;
        vec.push_back(nt2);
        vec.push_back(nt3);
        std::map<size_t, std::vector<size_t>> mp1;
        mp1[nt1] = vec;
        vericomp_data[image_idx] = mp1;
    }
    else{
        auto mp1 = vericomp_data[image_idx];
        auto it1 = mp1.find(nt1);
        if(it1 == mp1.end()){
            std::vector<size_t> vec;
            vec.push_back(nt2);
            vec.push_back(nt3);
            mp1[nt1] = vec;
            vericomp_data[image_idx] = mp1;
        }
    }
}

std::map<size_t, std::map<size_t, std::vector<size_t>>> parse_file(std::string &file_path){
    std::map<size_t, std::map<size_t, std::vector<size_t>>> vericomp_data;
    size_t image_idx;
    size_t last_layer_nt_idx;
    size_t nt1;
    std::fstream newfile;
    newfile.open(file_path, std::ios::in);
    if(newfile.is_open()){
        std::string tp;
        std::string acc = "";
        while(getline(newfile, tp)){
            if(tp != ""){
                size_t counter = 1;
                for(size_t i=0; i<tp.size(); i++){
                    if(std::isspace(tp[i]) and acc != ""){
                        size_t val = std::stoul(acc);
                        if(counter == 1){
                            image_idx = val;
                        }
                        else if(counter == 2){
                            last_layer_nt_idx = val;
                        }
                        else if(counter == 3){
                            nt1 = val;
                        }
                        else if(counter > 3){
                            update_data(vericomp_data, image_idx, last_layer_nt_idx, nt1, val);
                        }
                        counter++;
                        acc = "";
                    }
                    else if(std::isspace(tp[i])){
                        continue;
                    }
                    else if(tp[i] == '_'){
                        acc = "";
                    }
                    else{
                        acc += tp[i];
                    }
                }
            }
        }
    }
    else{
        assert(0 && "Wrong file path");
    }

    return vericomp_data;
}

void parse_file_and_update_bounds(Network_t* net, std::string &file_path){
    file_path = "/home/u1411251/Downloads/bounds_66.txt";
    size_t NUM_LAYERS = 16;
    size_t upto_layer_index = 2*(NUM_LAYERS -1);
    size_t image_index;
    size_t layer_idx;
    size_t nt_idx;
    double lb=-INFINITY;
    double ub=INFINITY;
    std::fstream newfile;
    newfile.open(file_path, std::ios::in);
    if(newfile.is_open()){
        std::string tp;
        std::string acc = "";
        while(getline(newfile, tp)){
            if(tp != ""){
                size_t counter = 0;
                for(size_t i=0; i<tp.size(); i++){
                    if(std::isspace(tp[i]) && acc != ""){
                        if(counter == 0){
                            image_index = std::stoul(acc);
                        }
                        if(counter == 1){
                            layer_idx = std::stoul(acc);
                            layer_idx = layer_idx - 1;
                        }
                        else if(counter == 2){
                            nt_idx = std::stoul(acc);
                        }
                        else if(counter == 3){
                            ub = std::stod(acc);
                        }
                        else if(counter == 4){
                            if(image_index == Configuration_deeppoly::image_index && layer_idx <= upto_layer_index){
                                lb = std::stod(acc);
                                std::cout<<"Check.. : "<<image_index<<" "<<layer_idx+1<<" "<<nt_idx<<" "<<ub<<" "<<lb<<std::endl;
                                Layer_t* layer = net->layer_vec[layer_idx];
                                Neuron_t* nt = layer->neurons[nt_idx];
                                nt->lb = -lb;
                                nt->ub = ub;
                            }
                        }
                        counter++;
                        acc = "";
                    }
                    else if(std::isspace(tp[i])){
                        continue;
                    }
                    else{
                        acc += tp[i];
                    }
                }
            }
        }
    }
}

drefine_status is_verified_by_vericomp(Network_t* net){
    size_t num_top_neurons = 0;
    if(num_top_neurons <= 0){
        return UNKNOWN;
    }
    std::string file_path = "/home/u1411251/Documents/tools/VeriNN/deep_refine/output_open_relu_2.txt";
    std::vector<std::vector<double>> imp_vec_vec = compute_pre_imp(net);
    std::vector<double> last_layer_imp_vec = imp_vec_vec[0];
    std::vector<size_t> sorted_vec = imp_based_sorted_index(last_layer_imp_vec);
    std::map<size_t, std::map<size_t, std::vector<size_t>>> vericomp_data = parse_file(file_path);
    std::cout<<"Size of map: "<<vericomp_data.size()<<std::endl;
    Layer_t* layer = net->layer_vec[0];
    for(size_t i=0; i< sorted_vec.size() && i<num_top_neurons; i++){
        size_t nt_idx = sorted_vec[i];
        std::map<size_t, std::vector<size_t>> vericomp_data_map = vericomp_data[Configuration_deeppoly::image_index];
        std::vector<size_t> mark_neurons = vericomp_data_map[nt_idx];
        for(size_t marked_index : mark_neurons){
            layer->neurons[marked_index]->is_marked = true;
        }
    }

    size_t count = 0;
    for(Neuron_t* nt : layer->neurons){
        if(nt->is_marked){
            count++;
        }
    }
    if(count == 0){
        return UNKNOWN;
    }
    std::cout<<"#Comp node: "<<count<<std::endl;
    bool is_image_verified = is_image_verified_by_milp(net);
    if(is_image_verified){
        return VERIFIED;
    }
    return UNKNOWN;
}


// drefine_status is_verified_by_vericomp1(Network_t* net){
//     size_t num_nt_to_select = 10;
//     std::vector<int> layer_idx = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1};
//     std::vector<size_t> nt_index = {39, 96, 46, 91, 75, 14, 34, 95, 63, 8, 50, 61};
//     assert(layer_idx.size() == nt_index.size() && "some mistakes in taking vericomp important neurons\n");
//     std::cout<<"All fine\n"<<std::endl;
//     for(size_t i=0; (i<layer_idx.size() && i<num_nt_to_select); i++){
//         int l_idx = layer_idx[i];
//         size_t n_idx = nt_index[i];
//         net->layer_vec[l_idx-1]->neurons[n_idx]->is_marked = true;
//     }

//     bool is_verified = is_image_verified_by_milp(net);
//     if (is_verified){
//         return VERIFIED;
//     }
//     else{
//         return UNKNOWN;
//     }
// }
