#include "network.hh" // network.hh is included in parser.hh
#include<stdio.h>
#include<fstream>
#include<vector>


Network_t::Network_t(){
    printf("\nNetwork constructor called\n");
}

std::vector<std::string> parse_string(std::string ft){
    char delimeter = ',';
    std::vector<std::string> vec;
    std::string acc = "";
    for(int i=0; i<ft.size();i++){
        if(ft[i] == delimeter){
            vec.push_back(acc);
            acc = "";
        }
        else{
            acc += ft[i];
        }
    }
    if(acc != ""){
        vec.push_back(acc);
    }     
    return vec;
}

void parse_string_to_xarray(Network_t* net, std::string weights, bool is_bias, size_t layer_index){
    std::vector<double> weight_vec;
    char comma = ',';
    char left_brac = '[';
    char right_brac = ']';
    std::vector<std::string> vec;
    std::string acc = "";
    for(int i=0; i<weights.size();i++){
        if(weights[i] == comma || weights[i] == right_brac){
            if(acc != ""){
                double val = std::stod(acc);
                weight_vec.push_back(val);
                acc = "";    
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

    std::tuple<size_t,size_t,size_t> t = net->layer_vec[layer_index]->w_shape;
    if(is_bias){
        std::vector<size_t> shape = {std::get<2>(t)};
        net->layer_vec[layer_index]->b = xt::adapt(weight_vec, shape);
    }
    else{
        std::vector<size_t> shape = {std::get<1>(t), std::get<2>(t)};
        net->layer_vec[layer_index]->w = xt::adapt(weight_vec, shape);
    }
}

void init_expr_coeffs(Neuron_t* nt, std::vector<std::string> &coeffs, bool is_upper){
    if(is_upper){
        for(int i = 1; i < coeffs.size(); i++){
            nt->ucoeffs.push_back(std::stod(coeffs[i]));
        }
    }
    else{
        for(int i = 1; i < coeffs.size(); i++){
            nt->lcoeffs.push_back(std::stod(coeffs[i]));
        }
    }
    
}

void init_input_layer(z3::context &c, Network_t* net){
    Layer_t* input_layer = new Layer_t();
    for(size_t i=0;i<net->input_dim;i++){
        Neuron_t* nt = new Neuron_t();
        nt->neuron_index = i;
        std::string nt_str = "i_"+std::to_string(i);
        nt->nt_z3_var = c.real_const(nt_str.c_str());
        input_layer->neurons.push_back(nt);
    }
    net->input_layer = input_layer;
}

void init_network(z3::context &c, Network_t* net, std::string file_path){
    std::fstream newfile;
    newfile.open(file_path, std::ios::in);
    int layer_index = 0;
    int neuron_index = 0;
    if(newfile.is_open()){
        std::string tp;
        Layer_t* curr_layer;
        Neuron_t* curr_neuron;
        while (getline(newfile, tp)){
            if(tp != ""){
                std::vector<std::string> tokens =  parse_string(tp);
                if(tokens[0] == "inputdim"){
                    net->input_dim = stoi(tokens[1]);
                }
                else if(tokens[0] == "layer"){
                    curr_layer = new Layer_t();
                    layer_index = stoi(tokens[1]);
                    bool is_activation;
                    if(tokens[2] == "1"){
                        is_activation = true;
                    }
                    else{
                        is_activation = false;
                    }
                    std::string activation = "";
                    curr_layer->activation = activation;
                    curr_layer->is_activation = is_activation;
                    curr_layer->layer_index = layer_index;
                    curr_layer->dims=stoi(tokens[3]);//number of neurons in current layer
                    net->layer_vec.push_back(curr_layer);
                    net->numlayers++;
                }
                else if(tokens[0] == "neuron"){
                    curr_neuron = new Neuron_t();
                    neuron_index = stoi(tokens[1]);
                    curr_neuron->neuron_index = neuron_index;
                    curr_neuron->lb = tokens[2];
                    curr_neuron->ub = tokens[3];
                    curr_layer->neurons.push_back(curr_neuron);
                }
                else if(tokens[0] == "upper"){
                    init_expr_coeffs(curr_neuron,tokens,true);
                }
                else if(tokens[0] == "lower"){
                    init_expr_coeffs(curr_neuron,tokens,false);
                }
            }
        }

        net->output_dim = net->layer_vec.back()->dims;
        
    }
    else{
        assert(0 && "Not able to open input file!!");
    }

    init_input_layer(c,net);
    set_weight_dims(net);
}



void set_weight_dims(Network_t* net){
    for(int i=0;i<net->numlayers;i++){
        size_t t0 = 0;
        Layer_t* curr_layer = net->layer_vec[i];
        if(!curr_layer->is_activation){
            if(i == 0){
                size_t t1 = net->input_dim;
                size_t t2 = curr_layer->dims;
                net->wt_shapes.push_back(std::make_tuple(t0,t1,t2));
                curr_layer->w_shape = std::make_tuple(t0,t1,t2);
            }
            else{
                size_t t1 = net->layer_vec[i-1]->dims; //prevlayer's dimension
                size_t t2 = curr_layer->dims;
                net->wt_shapes.push_back(std::make_tuple(t0,t1,t2));
                curr_layer->w_shape = std::make_tuple(t0,t1,t2);
            }    
        }   
    }
}

void init_net_weights(Network_t* net, std::string &filepath){
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
                    parse_string_to_xarray(net, tp, false, layer_index);
                    getline(newfile, tp);
                    parse_string_to_xarray(net, tp, true, layer_index);
                    layer_index = layer_index + 2;
                }
            }
        }
    }
}

void Neuron_t::print_neuron(){
    std::cout<<this->nt_z3_var<<", upper: "<<this->z_uexpr<<", lower: "<<this->z_lexpr<<"\n";
}

void Layer_t::print_layer(){
    std::cout<<"Layer index: "<<this->layer_index<<"\n";
    for(auto nt : this->neurons){
        nt->print_neuron();
    }
}

void Network_t::print_network(){
    this->input_layer->print_layer();
    for(auto layer : this->layer_vec){
        layer->print_layer();
    }
}

void Network_t::forward_propgate_one_layer(size_t layer_index, 
                                           xt::xarray<double> &inp){
    Layer_t* layer = this->layer_vec[layer_index];
    if(layer->is_activation){
        layer->res = xt::maximum(inp, 0);
    }
    else{
        xt::xarray<double> matmul = xt::linalg::dot(inp, layer->w);
        layer->res = matmul + layer->b;
    }

}

void Network_t::forward_propgate_network(size_t layer_index, 
                              xt::xarray<double> &inp){
    bool is_first = true;
    for(size_t i = layer_index; i < this->numlayers; i++){
        if(is_first){
            this->forward_propgate_one_layer(i,inp);
            is_first = false;
        }
        else{
            this->forward_propgate_one_layer(i, this->layer_vec[i-1]->res);
        }
    }
}

void parse_image_string_to_xarray_one(Network_t* net, std::string &image_str){
    char delimeter = ',';
    std::vector<double> vec;
    std::string acc = "";
    for(int i=1; i<image_str.size();i++){
        if(image_str[i] == delimeter){
            std::cout<<acc<<std::endl;
            std::cout<<acc.size()<<std::endl;
            //double val = std::stod(acc);
            //vec.push_back(val);
            acc = "";
        }
        else if(!std::isspace(image_str[i])){
            acc += image_str[i];
        }
    }
    if(acc != ""){
        std::cout<<acc<<std::endl;
        std::cout<<acc.size()<<std::endl;
        //double val = std::stod(acc);
        //vec.push_back(val);
    }
    std::vector<size_t> shape = {net->input_dim};
    net->im = xt::adapt(vec,shape);
}

void parse_image_string_to_xarray(Network_t* net, std::string &image_path){
    std::fstream newfile;
    newfile.open(image_path, std::ios::in);
    if(newfile.is_open()){
        std::string tp;
        while (getline(newfile, tp)){
            if(tp != ""){
                std::cout<<tp<<std::endl;
                parse_image_string_to_xarray_one(net,tp);
            }
        }
    }
}







