#include "network.hh"
#include<stdio.h>
#include<fstream>
#include<vector>




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

layer_t* create_layer(bool is_activation,std::string activation){
    layer_t *layer = (layer_t*)malloc(sizeof(layer_t));
    layer->is_activation = is_activation;
    return layer;
}

neuron_t* create_neuron(std::string lb, std::string ub){
    neuron_t *neuron = (neuron_t*)malloc(sizeof(neuron_t));
    neuron->lb = lb;
    neuron->ub = ub;
    return neuron;
}

void add_expr(neuron_t* nt, std::vector<std::string> &coeffs, bool is_upper){
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

void init_network(fppoly_t *fp, std::string file_path){
    std::fstream newfile;
    newfile.open(file_path, std::ios::in);
    int layer_index = 0;
    int neuron_index = 0;
    if(newfile.is_open()){
        std::string tp;
        layer_t* curr_layer;
        neuron_t* curr_neuron;
        while (getline(newfile, tp)){
            if(tp != ""){
                std::vector<std::string> tokens =  parse_string(tp);
                if(tokens[0] == "layer"){
                    layer_index = stoi(tokens[1]);
                    bool is_activation;
                    if(tokens[2] == "1"){
                        is_activation = true;
                    }
                    else{
                        is_activation = false;
                    }
                    std::string activation = "";
                    curr_layer = create_layer(is_activation,activation);
                    fp->layer_map.push_back(curr_layer);
                }
                else if(tokens[0] == "neuron"){
                    neuron_index = stoi(tokens[1]);
                    std::string lb = tokens[2];
                    std::string ub = tokens[3];
                    curr_neuron = create_neuron(lb,ub);
                    curr_layer->neurons.push_back(curr_neuron);
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

int main(){
    std::string filepath = "/home/u1411251/Documents/Phd/tools/ERAN/tf_verify/fppolyForward.txt";
    fppoly_t *fp = (fppoly_t*)malloc(sizeof(fppoly_t));
    init_network(fp,filepath);
    return 0;
}




