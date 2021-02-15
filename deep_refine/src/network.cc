#include "network.hh"
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
                if(tokens[0] == "layer"){
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
                    curr_layer->dims=0;
                    net->layer_vec.push_back(curr_layer);
                }
                else if(tokens[0] == "neuron"){
                    curr_neuron = new Neuron_t();
                    neuron_index = stoi(tokens[1]);
                    curr_neuron->neuron_index = neuron_index;
                    curr_neuron->lb = tokens[2];
                    curr_neuron->ub = tokens[3];
                    curr_layer->neurons.push_back(curr_neuron);
                    curr_layer->dims++;
                }
                else if(tokens[0] == "upper"){
                    init_expr_coeffs(curr_neuron,tokens,true);
                }
                else if(tokens[0] == "lower"){
                    init_expr_coeffs(curr_neuron,tokens,false);
                }
            }
        }
        
    }
    else{
        assert(0 && "Not able to open input file!!");
    }

    init_input_layer(c,net);
}

void set_predecessor_layer_activation(z3::context& c, Layer_t* layer, Layer_t* prev_layer){
    for(int i=0;i<layer->neurons.size();i++){
        Neuron_t* nt = layer->neurons[i];
        std::string nt_str = "nt_"+std::to_string(layer->layer_index)+","+std::to_string(nt->neuron_index);
        nt->nt_z3_var = c.real_const(nt_str.c_str());
        nt->pred_neurons.push_back(prev_layer->neurons[i]);
    }
}

void set_predecessor_layer_matmul(z3::context& c, Layer_t* layer, Layer_t* prev_layer){
    for(int i=0; i<layer->neurons.size(); i++){
        Neuron_t* nt = layer->neurons[i];
        std::string nt_str = "nt_"+std::to_string(layer->layer_index)+","+std::to_string(nt->neuron_index);
        nt->nt_z3_var = c.real_const(nt_str.c_str());
        nt->pred_neurons.assign(prev_layer->neurons.begin(), prev_layer->neurons.end());
    }
}

void set_predecessor_and_z3_var(z3::context &c, Network_t* net){
    Layer_t* prev_layer = new Layer_t();
    for(auto layer:net->layer_vec){
        if(layer->layer_index == 0){
            prev_layer = net->input_layer;
        }
        else{
            prev_layer = net->layer_vec[layer->layer_index - 1];
        }
        if(layer->is_activation){
            set_predecessor_layer_activation(c,layer,prev_layer);
        }
        else{
            set_predecessor_layer_matmul(c,layer,prev_layer);
        }
    }
}

z3::expr get_expr_from_double(z3::context &c, double item){
    std::string item_str = std::to_string(item);
    return c.real_val(item_str.c_str());
}

void init_z3_expr_neuron(z3::context &c, Neuron_t* nt){
    z3::expr sum1 = get_expr_from_double(c,nt->ucoeffs.back());
    z3::expr sum2 = get_expr_from_double(c,nt->lcoeffs.back());

    for(size_t i=0; i<nt->pred_neurons.size();i++){
        z3::expr u_coef = get_expr_from_double(c,nt->ucoeffs[i]);
        z3::expr l_coef = get_expr_from_double(c, nt->lcoeffs[i]);
        sum1 = sum1 + u_coef*nt->pred_neurons[i]->nt_z3_var;
        sum2 = sum2 + l_coef*nt->pred_neurons[i]->nt_z3_var;
    }
    nt->z_uexpr = nt->nt_z3_var <= sum1;
    nt->z_lexpr = nt->nt_z3_var >= sum2;
}


void init_z3_expr_layer(z3::context &c, Layer_t* layer){   
    z3::expr t_expr = c.bool_val(true);
    for(auto nt:layer->neurons){
        init_z3_expr_neuron(c,nt);
        t_expr = t_expr && nt->z_uexpr && nt->z_lexpr;
    }
    layer->layer_expr = t_expr;
}

void init_z3_expr(z3::context &c, Network_t *net){
    for(auto layer : net->layer_vec){
        if(layer->layer_index != 0){
            init_z3_expr_layer(c,layer);
        }
    }
}






