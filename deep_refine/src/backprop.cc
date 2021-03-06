#include "backprop.hh"

void back_substitute_neuron(z3::context &c, Neuron_t* nt){
    size_t upper_size = nt->ucoeffs.size()-1;//last element is just a constant d, i.e. cx+d
    Z3_ast src_u [upper_size];
    Z3_ast dest_u [upper_size];
    Z3_ast src_l [upper_size];
    Z3_ast dest_l [upper_size];
    size_t u_count = 0;
    size_t l_count = 0;
    for(int i=0; i < upper_size; i++){
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
     z3::expr t_expr = c.bool_val(true);
    for(auto nt:layer->neurons){
        back_substitute_neuron(c,nt);
        t_expr = t_expr && nt->nt_z3_var <= nt->z_uexpr && nt->nt_z3_var >= nt->z_lexpr;
    }
    t_expr.simplify();
}

void back_substitute(Network_t* net){
    for(auto layer:net->layer_vec){
        if(layer->layer_index != 0){
            Layer_t* prev_layer = net->layer_vec[layer->layer_index -1];

        }
    }
}


// int main(){
//     std::string filepath = "/home/u1411251/Documents/Phd/tools/ERAN/tf_verify/fppolyForward.txt";
//     Network_t* net = new Network_t();
//     z3::context c;
//     init_network(c,net,filepath);
//     set_predecessor_and_z3_var(c,net);
//     init_z3_expr(c,net);
//     printf("\nCheck..\n");
//     return 0;
// }
