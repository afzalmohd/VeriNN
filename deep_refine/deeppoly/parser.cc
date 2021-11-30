#include "parser.hh"
#include "configuration.hh"
#include<fstream>


void init_network(Network_t* net, std::string &filepath){
    std::fstream newfile;
    newfile.open(filepath, std::ios::in);
    if(newfile.is_open()){
        std::string tp;
        std::string relu_str = "ReLU";
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
            }
        }
        net->input_layer = create_input_layer(Configuration::input_dim);
        net->numlayers = net->layer_vec.size();
        net->input_dim = net->input_layer->dims;
        net->output_dim = net->layer_vec.back()->dims;
        pred_layer_linking(net);
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
        int image_counter = 0;
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
    char delimeter = ',';
    std::vector<double> vec;
    std::string acc = "";
    size_t pixel_counter = 0;
    bool is_first = true;
    for(size_t i=0; i<image_str.size();i++){
        if(image_str[i] == delimeter){
            if(acc != ""){
                if(is_first){
                    //std::cout<<acc<<std::endl;
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