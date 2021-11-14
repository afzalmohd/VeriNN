#include "network.hh"
#include<stdio.h>
#include<fstream>
#include<vector>
#include "z3expr.hh"
#include "backprop.hh"

z3::expr get_expr_from_double(z3::context &c, double item){
    std::string item_str = std::to_string(item);
    return c.real_val(item_str.c_str());
}

bool is_number(std::string s){
    for(size_t i = 0; i<s.size(); i++){
        if(!(std::isdigit(s[i]) || s[i] == '.' || s[i] == '-')){
            return false;
        }
    }
    return !s.empty();
}

std::vector<std::string> parse_string(std::string ft){
    char delimeter = ',';
    std::vector<std::string> vec;
    std::string acc = "";
    for(size_t i=0; i<ft.size();i++){
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
    for(size_t i=0; i<weights.size();i++){
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
        auto aisehi = net->layer_vec[layer_index]->b.shape();
        
        std::cout<<"Bias matrix shape: " <<xt::adapt(net->layer_vec[layer_index]->b.shape())<<std::endl;
    }
    else{
        std::vector<size_t> shape = {std::get<2>(t), std::get<1>(t)};
        xt::xarray<double> temp = xt::adapt(weight_vec, shape);
        net->layer_vec[layer_index]->w = xt::transpose(temp);
        std::cout<<"Weight matrix shape: "<<xt::adapt(net->layer_vec[layer_index]->w.shape())<<std::endl;
    }
}

void init_expr_coeffs(Neuron_t* nt, std::vector<std::string> &coeffs, bool is_upper){
    if(is_upper){
        for(size_t i = 1; i < coeffs.size(); i++){
            nt->ucoeffs.push_back(std::stod(coeffs[i]));
        }
    }
    else{
        for(size_t i = 1; i < coeffs.size(); i++){
            nt->lcoeffs.push_back(std::stod(coeffs[i]));
        }
    }
    
}

void init_input_layer(z3::context &c, Network_t* net){
    Layer_t* input_layer = new Layer_t(c);
    for(size_t i=0;i<net->input_dim;i++){
        Neuron_t* nt = new Neuron_t(c);
        nt->neuron_index = i;
        std::string nt_str = "i_"+std::to_string(i);
        //std::cout<<nt_str<<std::endl;
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
                    curr_layer = new Layer_t(c);
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
                    curr_neuron = new Neuron_t(c);
                    neuron_index = stoi(tokens[1]);
                    curr_neuron->neuron_index = neuron_index;
                    if(is_number(tokens[2])){
                        curr_neuron->lb = std::stod(tokens[2]);
                    }
                    if(is_number(tokens[3])){
                        curr_neuron->ub = std::stod(tokens[3]);
                    }

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
    for(size_t i=0;i<net->numlayers;i++){
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
                    parse_string_to_xarray(net, tp, false, layer_index); //weight matrix
                    getline(newfile, tp);
                    parse_string_to_xarray(net, tp, true, layer_index); //bias matrix
                    layer_index = layer_index + 2;
                }
            }
        }
    }
}

void Neuron_t::print_neuron(){
    std::cout<<this->lb<<" <= "<<this->nt_z3_var<<" <= "<<this->ub<<"\n";
    std::cout<<this->nt_z3_var<<", upper: "<<this->z_uexpr<<", lower: "<<this->z_lexpr<<"\n";
}

void Layer_t::print_layer(){
    std::cout<<"Layer index: "<<this->layer_index<<"\n";
    //std::cout<<"Number of neuron: "<<this->vars.size()<<std::endl;
    // std::cout<<"weight: "<<this->w<<"\n";
    // std::cout<<"biases: "<<this->b<<"\n";
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

void construct_and_execute_image(size_t layer_index, Network_t* net){
    Layer_t* layer = net->layer_vec[layer_index];
    std::vector<double> sat_assign;
    for(auto nt:layer->neurons){
        sat_assign.push_back(nt->back_prop_val);
    }
    std::vector<std::size_t> shape = {layer->dims};
    xt::xarray<double> gen_image = xt::adapt(sat_assign, shape);
    //std::cout<<"Layer index is: "<<layer_index<<", shape: "<<layer->dims<<std::endl;
    net->forward_propgate_network(layer_index+1, gen_image);
    std::cout<<"Layer index: "<<layer_index<<", output is: "<<net->layer_vec.back()->res<<std::endl;
}

void parse_image_string_to_xarray_one(Network_t* net, std::string &image_str){
    char delimeter = ',';
    std::vector<double> vec;
    std::string acc = "";
    //std::cout<<image_str<<std::endl;
    for(size_t i=1; i<image_str.size();i++){
        if(image_str[i] == delimeter){
            if(acc != ""){
                double val = std::stod(acc);
                vec.push_back(val);
                acc = "";
            }
        }
        else if(!std::isspace(image_str[i])){
            acc += image_str[i];
        }
    }
    if(acc != ""){
        double val = std::stod(acc);
        vec.push_back(val);
    }
    std::vector<size_t> shape = {net->input_dim};
    net->im = xt::adapt(vec,shape) / 255;
    //std::cout<<net->im<<std::endl;
}

void parse_image_string_to_xarray(Network_t* net, std::string &image_path){
    std::fstream newfile;
    newfile.open(image_path, std::ios::in);
    if(newfile.is_open()){
        std::string tp;
        int image_counter = 0;
        while (getline(newfile, tp)){
            if(tp != ""){
                if(image_counter < 1){
                    parse_image_string_to_xarray_one(net,tp);
                    image_counter++;
                }
                else{
                    break;
                }
            }
        }
    }
}

void create_prop(z3::context &c, Network_t* net){
    net->forward_propgate_network(0, net->im);
    Layer_t* last_layer = net->layer_vec.back();
    xt::xarray<std::size_t> out = xt::argmax(last_layer->res);
    size_t cl = out[0];
    z3::expr prop = c.bool_val(true);
    for(size_t i=0; i<last_layer->neurons.size(); i++){
        if(i != cl){
            prop = prop && (last_layer->neurons[cl]->nt_z3_var >= last_layer->neurons[i]->nt_z3_var);
        }
    }
    net->prop_expr = prop;
    //printf("\nCheck..\n");
    //std::cout<<net->prop_expr<<std::endl;
}

void init_input_box(z3::context &c, Network_t* net){
    z3::expr t_expr = c.bool_val(true);
    Layer_t* inp_layer = net->input_layer;
    for(size_t i = 0; i < inp_layer->neurons.size(); i++){
        Neuron_t* nt = inp_layer->neurons[i];
        double upper_bound = net->im[i] + net->epsilon;
        double lower_bound = net->im[i] - net->epsilon;
        if(upper_bound > 1.0){
            upper_bound = 1.0;
        }
        if(lower_bound < 0.0){
            lower_bound = 0.0;
        }
        std::string upper_str = std::to_string(upper_bound);
        std::string lower_str = std::to_string(lower_bound);
        if(net->is_my_test){
            upper_str = "1.0";
            lower_str = "-1.0";
            nt->ub = 1.0;
            nt->lb = -1.0;
            t_expr = t_expr && nt->nt_z3_var <= c.real_val(upper_str.c_str()) && nt->nt_z3_var >= c.real_val(lower_str.c_str());
        }
        else{
            nt->lb = lower_bound;
            nt->ub = upper_bound;
            t_expr = t_expr && nt->nt_z3_var <= c.real_val(upper_str.c_str()) && nt->nt_z3_var >= c.real_val(lower_str.c_str());
        }
        
    }
    inp_layer->b_expr = t_expr;
    //std::cout<<net->input_layer->layer_expr<<std::endl;
}


int find_refine_nodes(int num_params, char* params[]) {

    Configuration::init_options(num_params, params);
    if(Configuration::vm.count("help")){
        return 0;
    }

    z3::context c;
    Network_t* net = new Network_t(c);
    time_t curr_time = time(NULL);
    init_network(c,net,Configuration::abs_out_file_path);
    init_net_weights(net, Configuration::net_path);
    if(Configuration::is_small_ex){
        net->im = {117,211};
        net->im = net->im/255;
        std::cout<<"Image: "<<net->im<<std::endl;
    }
    else{
        parse_image_string_to_xarray(net, Configuration::dataset_path);
    }
    
    net->forward_propgate_network(0,net->im);
    std::cout<<net->layer_vec.back()->res<<std::endl;
    std::cout<<"Time in parser: "<<time(NULL) - curr_time<<std::endl;
    set_predecessor_and_z3_var(c,net);
    curr_time = time(NULL);
    init_input_box(c,net);
    init_z3_expr(c,net);
    std::cout<<"Time in z3expr init: "<<time(NULL) - curr_time<<std::endl;  
    create_prop(c,net); 
    affine_expr_init(c,net);
    
    //net->print_network();
    curr_time = time(NULL);
    //prop_back_propogate(c,net);
    check_sat_output_layer(c,net);
    std::cout<<"Time to check satisfiability: "<<time(NULL) - curr_time<<std::endl;
    // for(auto layer:net->layer_vec){
    //     std::cout<<"Layer index is: "<<layer->layer_index<<", neurons: "<<layer->dims<<std::endl;
    // }
    printf("\nEnded\n");
    return 0;
}






