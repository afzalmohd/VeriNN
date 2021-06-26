#include "backprop.hh"

void back_substitute_neuron(z3::context &c, Neuron_t* nt){
    size_t upper_size = nt->ucoeffs.size()-1;//last element is just a constant d, i.e. cx+d
    Z3_ast src_u [upper_size];
    Z3_ast dest_u [upper_size];
    Z3_ast src_l [upper_size];
    Z3_ast dest_l [upper_size];
    size_t u_count = 0;
    size_t l_count = 0;
    for(size_t i=0; i < upper_size; i++){
        Neuron_t* pred_nt = nt->pred_neurons[i];
        if(nt->ucoeffs[i] > 0.0){
            src_u[u_count] = pred_nt->nt_z3_var;
            dest_u[u_count] = pred_nt->z_uexpr;
            u_count++;
        }
        else if(nt->ucoeffs[i] < 0.0){
            src_u[u_count] = pred_nt->nt_z3_var;
            dest_u[u_count] = pred_nt->z_lexpr;
            u_count++;
        }
        if(nt->lcoeffs[i] > 0.0){
            src_l[l_count] = pred_nt->nt_z3_var;
            dest_l[l_count] = pred_nt->z_lexpr;
            l_count++;
        }
        else if(nt->lcoeffs[i] < 0.0){
            src_l[l_count] = pred_nt->nt_z3_var;
            dest_l[l_count] = pred_nt->z_uexpr;
            l_count++;
        }   
    }
    
    nt->z_uexpr = to_expr(c, Z3_substitute(c, nt->z_uexpr, u_count, src_u, dest_u));
    nt->z_lexpr = to_expr(c, Z3_substitute(c, nt->z_lexpr, l_count, src_l, dest_l));
    nt->z_uexpr.simplify();
    nt->z_lexpr.simplify();
}

void back_substitute_layer(z3::context& c, Layer_t* layer){
    printf("\nCheck..\n");
    z3::expr t_expr = c.bool_val(true);
    for(auto nt:layer->neurons){
        back_substitute_neuron(c,nt);
        t_expr = t_expr && nt->nt_z3_var <= nt->z_uexpr && nt->nt_z3_var >= nt->z_lexpr;
    }
    t_expr.simplify();
}

void back_substitute(z3::context& c, Network_t* net){
    for(auto layer:net->layer_vec){
        if(layer->layer_index != 0){
            back_substitute_layer(c,layer);
        }
    }
}

void prop_back_propogate_layer(z3::context& c,z3::expr& neg_prop, Layer_t* layer){
    if(!layer->is_activation){
        z3::expr temp_expr = neg_prop;
        size_t arr_size = layer->dims;
        std::cout<<"Layer index: "<<layer->layer_index<<", dim: "<<arr_size<<", is affine"<<std::endl;
        Z3_ast src [arr_size];
        Z3_ast dest [arr_size];
        for(size_t i=0; i<arr_size; i++){
            Neuron_t* nt = layer->neurons[i];
            src[i] = nt->nt_z3_var;
            dest[i] = nt->affine_expr;
        }
        neg_prop = to_expr(c, Z3_substitute(c, temp_expr, arr_size, src, dest));
        neg_prop = neg_prop.simplify();
    }
    else{
        std::cout<<"Layer index: "<<layer->layer_index<<", dim: "<<layer->dims<<", is relu"<<std::endl;
        neg_prop = neg_prop && layer->c_expr && layer->b_expr;
        neg_prop = neg_prop.simplify();
    }
}



void prop_back_propogate(z3::context& c, Network_t* net){
    printf("\nCheck.\n");
    //affine_expr_init(c,net); Already called from main function
    z3::expr neg_prop = !net->prop_expr;
    //std::cout<<"Negation of prop: "<<neg_prop<<std::endl;
    for(int i = net->layer_vec.size()-1; i>=0; i--){
        Layer_t* layer = net->layer_vec[i];
        prop_back_propogate_layer(c, neg_prop, layer);
        //std::cout<<"Negation of prop after layer: "<<layer->layer_index<<std::endl;
        //std::cout<<neg_prop<<std::endl;
    }

    neg_prop = neg_prop && net->input_layer->b_expr;
    std::cout<<"Backsubstituition finished"<<std::endl;
    net->back_subs_prop = neg_prop;
    //std::cout<<"Input layer: "<<std::endl;
    //std::cout<<net->input_layer->b_expr<<std::endl;

    // for(int i=net->layer_vec.size()-1; i>=0; i--){
    //     if(i==4){
    //         for(auto nt : net->layer_vec[i]->neurons){
    //             if(nt->neuron_index == 0){
    //                 for(size_t j=0; j<nt->pred_neurons.size(); j++){
    //                     std::cout<<(net->layer_vec[i])->b(nt->neuron_index)<<std::endl;
    //                 }
    //             }
    //         }
    //     }
    // }
}

