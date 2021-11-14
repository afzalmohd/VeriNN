#include "network.hh"
#include "parser.hh"
#include "analysis.hh"
#include<vector>

int main(int argc, char* argv[]){
    Configuration::init_options(argc, argv);
    if(Configuration::vm.count("help")){
        return 0;
    }

    Network_t* net = new Network_t();
    init_network(net, Configuration::net_path);
    net->epsilon = Configuration::epsilon;
    execute_neural_network(net, Configuration::dataset_path);
    create_input_layer_expr(net);
    net->print_network();
    return 0;
}

void execute_neural_network(Network_t* net, std::string &image_path){
    size_t num_test = 1;
    for(int i=1; i <= num_test; i++){
        parse_input_image(net, image_path, i);
        net->forward_propgate_network(0, net->input_layer->res);
        std::cout<<"Predicted, actual label: ("<<xt::argmax(net->layer_vec.back()->res)<<","<<net->actual_label<<")"<<std::endl;
    }
}

void Expr_t::deep_copy(Expr_t* expr){
    this->coeff_inf.assign(expr->coeff_inf.begin(), expr->coeff_inf.end());
    this->coeff_sup.assign(expr->coeff_sup.begin(), expr->coeff_sup.end());
    this->cst_inf = expr->cst_inf;
    this->cst_sup = expr->cst_sup;
    this->size = expr->size;
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

void Neuron_t::print_neuron(){
    //std::cout<<this->lb<<" <= "<<this->nt_z3_var<<" <= "<<this->ub<<"\n";
    //std::cout<<this->nt_z3_var<<", upper: "<<this->z_uexpr<<", lower: "<<this->z_lexpr<<"\n";
    std::cout<<this->neuron_index<<", "<<this->lb<<", "<<this->ub<<std::endl;
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

