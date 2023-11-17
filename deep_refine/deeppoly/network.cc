#include "network.hh"
#include "parser.hh"
#include "analysis.hh"
#include "deeppoly_configuration.hh"


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

void Expr_t::deep_copy(Expr_t* expr){
    this->coeff_inf.assign(expr->coeff_inf.begin(), expr->coeff_inf.end());
    this->coeff_sup.assign(expr->coeff_sup.begin(), expr->coeff_sup.end());
    this->cst_inf = expr->cst_inf;
    this->cst_sup = expr->cst_sup;
    this->size = expr->size;
}


void Expr_t::print_expr(){
    for(size_t i=0; i<this->coeff_sup.size(); i++){
        std::cout<<this->coeff_sup[i]<<",";
    }
    std::cout<<this->cst_sup;
}


void Neuron_t::print_neuron(){
    std::cout<<"neuron,"<<this->neuron_index<<","<<-this->lb<<","<<this->ub<<std::endl;
}

void Layer_t::print_layer(){
    std::cout<<"layer,"<<this->layer_index<<","<<this->is_activation<<","<<this->dims<<std::endl;
     for(auto nt : this->neurons){
         nt->print_neuron();
     }
}

void Network_t::print_network(){
    std::cout<<"inputdim,"<<this->input_dim<<std::endl;
    for(auto layer : this->layer_vec){
            layer->print_layer();
    }
}

void reset_network(Network_t* net){
    for(auto layer: net->layer_vec){
        reset_layer(layer);
    }
}
void reset_layer(Layer_t* layer){
    for(size_t i=0; i < layer->neurons.size(); i++){
        Neuron_t* nt = layer->neurons[i];
        nt->~Neuron_t();
        nt->lb = INFINITY;
        nt->ub = INFINITY;
    }
}

void mark_layer_and_neurons(Layer_t* layer){
    if(Configuration_deeppoly::is_small_ex){
        if(layer->layer_index == 2){
            layer->is_marked = true;
            layer->neurons[0]->is_marked = true;
            layer->neurons[0]->is_active = true;
        }
    }
}

double round_off(double num, size_t prec){
    if(num == 0){
        return num;
    }
    double mult = pow(10.0f, float(prec));
    double val = mult*num + 0.5;
    double res = round(val) / mult;
    return res;
}

