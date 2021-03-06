#include "z3expr.hh" //network.hh included in z3expr.hh

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
        if(nt->ucoeffs[i] != 0.0){
            z3::expr u_coef = get_expr_from_double(c,nt->ucoeffs[i]);
            if(nt->ucoeffs[i] > 0.0){
                sum1 = sum1 + u_coef*(nt->pred_neurons[i]->z_uexpr);
            }
            else{
                sum1 = sum1 + u_coef*(nt->pred_neurons[i]->z_lexpr);
            }
        }
        if(nt->lcoeffs[i] != 0.0){
            z3::expr l_coef = get_expr_from_double(c, nt->lcoeffs[i]);
            if(nt->lcoeffs[i] > 0.0){
                sum2 = sum2 + l_coef*nt->pred_neurons[i]->z_lexpr;
            }
            else{
                sum2 = sum2 + l_coef*nt->pred_neurons[i]->z_uexpr;
            }
            
        }  
    }
    nt->z_uexpr = sum1.simplify();
    nt->z_lexpr = sum2.simplify();
}

void init_z3_expr_neuron_first(z3::context &c, Neuron_t* nt){
    z3::expr sum1 = get_expr_from_double(c,nt->ucoeffs.back());
    z3::expr sum2 = get_expr_from_double(c,nt->lcoeffs.back());

    for(size_t i=0; i<nt->pred_neurons.size();i++){
        if(nt->ucoeffs[i] != 0.0){
            z3::expr u_coef = get_expr_from_double(c,nt->ucoeffs[i]);
            sum1 = sum1 + u_coef*nt->pred_neurons[i]->nt_z3_var;
        }
        if(nt->lcoeffs[i] != 0.0){
            z3::expr l_coef = get_expr_from_double(c, nt->lcoeffs[i]);
            sum2 = sum2 + l_coef*nt->pred_neurons[i]->nt_z3_var;            
        }  
    }
    nt->z_uexpr = sum1;
    nt->z_lexpr = sum2;
}


void init_z3_expr_layer(z3::context &c, Layer_t* layer){   
    z3::expr t_expr = c.bool_val(true);
    for(auto nt:layer->neurons){
        init_z3_expr_neuron(c,nt);
        t_expr = t_expr && nt->nt_z3_var <= nt->z_uexpr && nt->nt_z3_var >= nt->z_lexpr;
    }
    layer->layer_expr = t_expr.simplify();
}

void init_z3_expr_layer_first(z3::context& c, Layer_t* layer){
    z3::expr t_expr = c.bool_val(true);
    for(auto nt:layer->neurons){
        init_z3_expr_neuron_first(c,nt);
        t_expr = t_expr && nt->nt_z3_var <= nt->z_uexpr && nt->nt_z3_var >= nt->z_lexpr;
    }
    layer->layer_expr = t_expr.simplify();
}

void init_z3_expr(z3::context &c, Network_t *net){
    for(auto layer : net->layer_vec){
        if(layer->layer_index == 0){
            init_z3_expr_layer_first(c,layer);
        }
        else{
            init_z3_expr_layer(c,layer);
        }
    }
}


int main(){
    std::string filepath = "/home/u1411251/Documents/Phd/tools/ERAN/tf_verify/fppolyForward.txt";
    std::string net_path = "/home/u1411251/Documents/Phd/tools/networks/mnist_relu_3_50_mod.tf";
    std::string dataset_path = "/home/u1411251/Documents/Phd/tools/ERAN/data/mnist_test.csv";
    Network_t* net = new Network_t();
    z3::context c;
    init_network(c,net,filepath);
    init_net_weights(net, net_path);
    printf("\nCheck..1\n");
    parse_image_string_to_xarray(net, dataset_path);
    net->forward_propgate_network(0,net->im);
    std::cout<<net->layer_vec.back()->res<<std::endl;
    //set_predecessor_and_z3_var(c,net);
    //init_z3_expr(c,net);
    //net->print_network();
    printf("\nCheck..\n");
    return 0;
}