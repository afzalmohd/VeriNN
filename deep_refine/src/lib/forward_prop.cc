#include "forward_prop.hh"

void init_z3_expr_neuron_forward(z3::context &c, Neuron_t* nt){
    
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

void init_z3_expr_neuron_first_forward(z3::context &c, Neuron_t* nt){
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


void init_z3_expr_layer_forward(z3::context &c, Layer_t* layer){   
    z3::expr t_expr = c.bool_val(true);
    for(auto nt:layer->neurons){
        init_z3_expr_neuron_forward(c,nt);
        t_expr = t_expr && nt->nt_z3_var <= nt->z_uexpr && nt->nt_z3_var >= nt->z_lexpr;
    }
    layer->c_expr = t_expr.simplify();
}

void init_z3_expr_layer_first_forward(z3::context& c, Layer_t* layer){
    z3::expr t_expr = c.bool_val(true);
    for(auto nt:layer->neurons){
        init_z3_expr_neuron_first_forward(c,nt);
        t_expr = t_expr && nt->nt_z3_var <= nt->z_uexpr && nt->nt_z3_var >= nt->z_lexpr;
    }
    layer->c_expr = t_expr.simplify();
    
}

void init_z3_expr_forward(z3::context &c, Network_t *net){
    for(auto layer : net->layer_vec){
        if(layer->layer_index == 0){
            init_z3_expr_layer_first_forward(c,layer);
        }
        else{
            init_z3_expr_layer_forward(c,layer);
        }
    }
}