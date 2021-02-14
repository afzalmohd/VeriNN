#include "network.hh"
#include<stdio.h>
#include<fstream>
#include<vector>


Fppoly_t::Fppoly_t(){
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

void add_expr(Neuron_t* nt, std::vector<std::string> &coeffs, bool is_upper){
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

void init_network(z3::context &c, Fppoly_t* fp, std::string file_path){
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
                    fp->layer_vec.push_back(curr_layer);
                }
                else if(tokens[0] == "neuron"){
                    curr_neuron = new Neuron_t();
                    neuron_index = stoi(tokens[1]);
                    curr_neuron->lb = tokens[2];
                    curr_neuron->ub = tokens[3];
                    curr_layer->neurons.push_back(curr_neuron);
                    curr_layer->dims++;
                }
                else if(tokens[0] == "upper"){
                    add_expr(curr_neuron,tokens,true);
                }
                else if(tokens[0] == "lower"){
                    add_expr(curr_neuron,tokens,false);
                }
            }
        }
        
    }
    else{
        assert(0 && "Not able to open input file!!");
    }
}

void init_z3_layer_expr(z3::context &c, Layer_t* layer){
    size_t ind = layer->layer_index;
    for(size_t i=0;i<layer->dims;i++){
        std::string x = "x_"+std::to_string(ind)+","+std::to_string(i);
        z3::expr ex_temp = c.real_const(x.c_str());
        layer->vars.push_back(ex_temp);
    }
}

void init_z3_expr(z3::context &c, Fppoly_t *fp){
    for(auto layer : fp->layer_vec){
        init_z3_layer_expr(c,layer);
    }
}

int main(){
    std::string filepath = "/home/u1411251/Documents/Phd/tools/ERAN/tf_verify/fppolyForward.txt";
    Fppoly_t* fp = new Fppoly_t();
    z3::context c;
    init_network(c,fp,filepath);
    init_z3_expr(c,fp);
    printf("\nCheck..\n");
    return 0;
}




