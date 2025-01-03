#include "parser.hh"
#include "deeppoly_configuration.hh"
#include<fstream>
#include<iostream>
#include<string>
#include<tuple>

void init_network(Network_t* net, std::string &filepath){
    if(filepath.substr(filepath.find_last_of(".") + 1) != "tf"){
        std::cout<<"Please provide tf format of network file"<<std::endl;
        assert(0);
    }
    std::fstream newfile;
    newfile.open(filepath, std::ios::in);
    if(newfile.is_open()){
        std::string tp;
        std::string relu_str = "ReLU";
        std::string matmul_str = "Gemm";
        size_t layer_index = 0;
        while (getline(newfile, tp)){
            if(tp != ""){
                if(tp == relu_str){
                    getline(newfile, tp);
                    Layer_t* layer = create_layer(false, "", "FC");
                    parse_string_to_xarray(layer, tp, false); //weight matrix
                    getline(newfile, tp);
                    parse_string_to_xarray(layer, tp, true); //bias matrix
                    
                    layer->layer_index = layer_index;
                    create_neurons_update_layer(layer);
                    net->layer_vec.push_back(layer);
                    layer_index++;
                    
                    Layer_t* relu_layer = create_layer(true, "ReLU", "FC"); //Creating relu layer seperately
                    relu_layer->dims = layer->dims;
                    relu_layer->layer_index = layer_index;
                    create_neurons_update_layer(relu_layer);
                    net->layer_vec.push_back(relu_layer);
                    layer_index++;
                }
                else if(tp == matmul_str){
                    getline(newfile, tp);
                    Layer_t* layer = create_layer(false, "", "FC");
                    parse_string_to_xarray(layer, tp, false); //weight matrix
                    getline(newfile, tp);
                    parse_string_to_xarray(layer, tp, true); //bias matrix
                    
                    layer->layer_index = layer_index;
                    create_neurons_update_layer(layer);
                    net->layer_vec.push_back(layer);
                    layer_index++;
                }
            }
        }
        net->input_layer = create_input_layer(Configuration_deeppoly::input_dim);
        net->numlayers = net->layer_vec.size();
        net->input_dim = net->input_layer->dims;
        net->output_dim = net->layer_vec.back()->dims;
        pred_layer_linking(net);
    }
    else{
        assert(0 && "Wrong network file path");
    }
}

void pred_layer_linking(Network_t* net){
    for(size_t i=0; i < net->layer_vec.size(); i++){
        Layer_t* layer = net->layer_vec[i];
        if(i == 0){
            layer->pred_layer = net->input_layer;
        }
        else{
            layer->pred_layer = net->layer_vec[i-1];
        }
    }
}

void parse_string_to_xarray(Layer_t* layer, std::string weights, bool is_bias){
    std::vector<double> weight_vec;
    char comma = ',';
    char left_brac = '[';
    char right_brac = ']';
    std::string acc = "";
    size_t rows = 0;
    size_t cols = 0;
    size_t count_cols = 0;
    size_t count_right_brac = 0;
    for(size_t i=0; i<weights.size();i++){
        if(weights[i] == comma){
            if(acc != ""){
                double val = std::stod(acc);
                weight_vec.push_back(val);
                acc = "";   
                count_cols++; 
                count_right_brac = 0;
            }    
        }
        else if(weights[i] == right_brac){
            if(acc != ""){
                double val = std::stod(acc);
                weight_vec.push_back(val);
                acc = "";
                count_cols++; 
                cols = count_cols;   
            } 
            count_right_brac++;
            if(count_right_brac == 1){
                count_cols = 0;
                rows++;
            }

        }
        else if(weights[i] != left_brac && !std::isspace(weights[i])){
            acc += weights[i];
        }
    }
    if(acc != ""){
         double val = std::stod(acc);
         weight_vec.push_back(val);
    }

    if(is_bias){
        std::vector<size_t> shape = {cols};
        layer->b = xt::adapt(weight_vec, shape);
        auto aisehi = layer->b.shape();
        std::cout<<"Bias matrix shape: " <<xt::adapt(layer->b.shape())<<std::endl;
        layer->dims = cols;
    }
    else{
        std::vector<size_t> shape = {rows, cols};
        xt::xarray<double> temp = xt::adapt(weight_vec, shape);
        layer->w = xt::transpose(temp);
        layer->w_shape = {cols, rows};
        std::cout<<"Weight matrix shape: "<<xt::adapt(layer->w.shape())<<std::endl;
    }
}

void parse_input_image(Network_t* net, std::string &image_path, size_t image_index){
    std::fstream newfile;
    newfile.open(image_path, std::ios::in);
    if(newfile.is_open()){
        std::string tp;
        size_t image_counter = 0;
        while (getline(newfile, tp)){
            if(tp != ""){
                if(image_counter == image_index){
                    parse_image_string_to_xarray_one(net,tp);
                    break;
                }
                else{
                    image_counter++;
                }
            }
        }
    }
}

void parse_image_string_to_xarray_one(Network_t* net, std::string &image_str){
    std::vector<double> vec;
    std::string acc = "";
    size_t pixel_counter = 0;
    bool is_first = true;
    for(size_t i=0; i<image_str.size();i++){
        if(image_str[i] == IMAGE_DELIMETER){
            if(acc != ""){
                if(is_first){
                    net->actual_label = std::stoi(acc);
                    is_first = false;
                    acc = "";
                }
                else{
                    double val = std::stod(acc);
                    vec.push_back(val);
                    acc = "";
                    pixel_counter++;
                }
            }
        }
        else if(!std::isspace(image_str[i])){
            acc += image_str[i];
        }
    }
    if(acc != ""){
        double val = std::stod(acc);
        vec.push_back(val);
        pixel_counter++;
    }
    assert(pixel_counter == net->input_dim && "Pixel count mismatch");
    std::vector<size_t> shape = {net->input_dim};
    net->input_layer->res = xt::adapt(vec,shape) / 255;
    // net->input_layer->res = xt::adapt(vec,shape);
}

Layer_t* create_layer(bool is_activation, std::string activation, std::string layer_type){
    Layer_t* layer = new Layer_t();
    layer->is_activation = is_activation;
    if(is_activation){
        layer->activation = activation;
    }
    else{
        layer->layer_type = layer_type;
    }
    return layer;
}

void create_neurons_update_layer(Layer_t* layer){
    for(size_t i=0; i < layer->dims;i++){
        Neuron_t* nt = new Neuron_t();
        nt->neuron_index = i;
        nt->layer_index = layer->layer_index;
        layer->neurons.push_back(nt);
    }
}

Layer_t* create_input_layer(size_t dim){
    Layer_t* layer = new Layer_t();
    layer->dims = dim;
    layer->layer_index = -1;
    create_neurons_update_layer(layer); 
    return layer;
}

std::tuple<size_t, double> get_index_bound(std::string& str_line){
    std::tuple<size_t, double> tup;
    size_t counter = 0;
    size_t t = 0;
    size_t nt_index = 0;
    double bound = 0.0;
    std::string acc = "";
    for(size_t i=0; i<str_line.size(); i++){
        if(std::isspace(str_line[i])){
            if(counter == 2){
                nt_index = std::stoul(acc);
            }
            acc = "";
            counter++;
        }
        else{
            if(counter == 2){
                t++;
                if(t >= 3){
                    acc += str_line[i];
                }
            }
            else if(counter == 3){
                if(str_line[i] != ')'){
                    acc += str_line[i];
                }
                else{
                    bound = std::stod(acc);
                }
            }
        }
    }
    
    tup = std::make_tuple(nt_index, bound);
    return tup;
}

double get_bound(std::string& str_line){
    std::tuple<size_t, double> tup;
    size_t counter = 0;
    double bound = 0.0;
    std::string acc = "";
    for(size_t i=0; i<str_line.size(); i++){
        if(std::isspace(str_line[i])){
            acc = "";
            counter++;
        }
        else if(counter == 3){
            if(str_line[i] != ')'){
                acc += str_line[i];
            }
            else{
                bound = std::stod(acc);
            }
        }
    }
    return bound;
}

size_t get_label(std::string& str_line){
    size_t counter = 0;
    std::string acc = "";
    size_t t = 0;
    size_t label=0;
    for(size_t i=0; i<str_line.size(); i++){
        if(std::isspace(str_line[i])){
            acc = "";
            counter++;
        }
        else if(counter == 7){
            t++;
            if(t >= 3){
                if(str_line[i] != ')'){
                    acc += str_line[i];
                }
                else{
                    label = std::stoul(acc);
                }
            }
        }
    }
    return label;
}

void parse_vnnlib_simplified_mnist(Network_t* net, std::string& file_path){
    Layer_t* input_layer = net->input_layer;
    std::fstream newfile;
    newfile.open(file_path, std::ios::in);
    if(newfile.is_open()){
        std::string tp;
        size_t nt_index = 0;
        std::string substr_ub = "(assert (<=";
        std::string substr_lb = "(assert (>=";
        std::string substr_actual_label = "(and (>=";
        double bound;
        while(getline(newfile, tp)){
            bool is_ub_found = tp.find(substr_ub) != std::string::npos;
            if(is_ub_found){
                bound = get_bound(tp);
                input_layer->neurons[nt_index]->ub = bound;
            }
            else{
                bool is_lb_found = tp.find(substr_lb) != std::string::npos;
                if(is_lb_found){
                    bound = get_bound(tp);
                    input_layer->neurons[nt_index]->lb = -bound;
                    nt_index++;
                }
                else{
                    bool is_out_label = tp.find(substr_actual_label) != std::string::npos;
                    if(is_out_label){
                        net->actual_label = get_label(tp);
                        net->pred_label = net->actual_label;
                        break;
                    }
                }
            }
        }
    }
}

std::vector<double> get_bounds_ab(std::string& str_line){
    std::vector<double> vec;
    size_t comma_counter = 0;
    std::string acc = "";
    double lb,ub;
    for(size_t i=0; i<str_line.size(); i++){
        if(str_line[i] == ','){
            comma_counter++;
            if(comma_counter == 2){
                lb = std::stod(acc);
            }
            acc = "";
        }
        else{
            acc += str_line[i];
        }
    }
    ub = std::stod(acc);
    vec.push_back(lb);
    vec.push_back(ub);

    return vec;
}

void bounds_parser(Network_t* net, std::string& file_path){
    std::fstream newfile;
    newfile.open(file_path, std::ios::in);
    size_t layer_index = 0;
    size_t counter = 0;
    if(newfile.is_open()){
        std::string tp;
        std::string layer_str = "Layer";
        Layer_t* curr_layer=NULL;
        size_t nt_index = 0;
        while(getline(newfile, tp)){
            if(tp != ""){
                bool is_layer_index = tp.find(layer_str) != std::string::npos;
                if(is_layer_index){
                    layer_index = 2*counter;
                    counter++;
                    nt_index = 0;
                }
                else{
                    curr_layer = net->layer_vec[layer_index];
                    std::vector<double> bounds = get_bounds_ab(tp);
                    Neuron_t* nt = curr_layer->neurons[nt_index];
                    nt->lb = -bounds[0];
                    nt->ub = bounds[1];

                    if(layer_index+1 < net->numlayers){
                        Layer_t* next_layer = net->layer_vec[layer_index+1];
                        Neuron_t* next_nt = next_layer->neurons[nt_index];
                        if(nt->ub <= 0){
                            next_nt->lb = 0;
                            next_nt->ub = 0;
                        }
                        else if(nt->lb <= 0){
                            next_nt->lb = nt->lb;
                            next_nt->ub = nt->ub;
                        }
                        else{
                            next_nt->lb = 0;
                            next_nt->ub = nt->ub;
                        }
                    }
                    nt_index++;
                }
            }
        }

    }
}