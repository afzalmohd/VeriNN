#include "z3expr.hh" //network.hh included in z3expr.hh
#include <ctime>
#include <fstream>
#include <iostream>

void set_z3_parameters(z3::context& c){
    c.set("ELIM_QUANTIFIERS", "true");
    z3::set_param("pp.decimal-precision", 5);
    z3::set_param("pp.decimal", true);
}

void set_predecessor_layer_activation(z3::context& c, Layer_t* layer, Layer_t* prev_layer){
    for(size_t i=0;i<layer->neurons.size();i++){
        Neuron_t* nt = layer->neurons[i];
        std::string nt_str = "nt_"+std::to_string(layer->layer_index)+","+std::to_string(nt->neuron_index);
        nt->nt_z3_var = c.real_const(nt_str.c_str());        
        nt->pred_neurons.push_back(prev_layer->neurons[i]);
    }
}

void set_predecessor_layer_matmul(z3::context& c, Layer_t* layer, Layer_t* prev_layer){
    for(size_t i=0; i<layer->neurons.size(); i++){
        Neuron_t* nt = layer->neurons[i];
        std::string nt_str = "nt_"+std::to_string(layer->layer_index)+","+std::to_string(nt->neuron_index);
        nt->nt_z3_var = c.real_const(nt_str.c_str());
        nt->pred_neurons.assign(prev_layer->neurons.begin(), prev_layer->neurons.end());
    }
}

void set_predecessor_and_z3_var(z3::context &c, Network_t* net){
    Layer_t* prev_layer;
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

void init_z3_expr(z3::context& c, Network_t* net){
    for(auto layer : net->layer_vec){
        init_z3_expr_layer(c, layer);
    }
    merged_constraints(c,net);
}

void init_z3_expr_layer(z3::context& c, Layer_t* layer){
    z3::expr t_expr = c.bool_val(true);
    z3::expr t_expr1 = c.bool_val(true);
    for(auto nt:layer->neurons){
        init_z3_expr_neuron(c,nt);
        t_expr = t_expr && nt->nt_z3_var <= nt->z_uexpr && nt->nt_z3_var >= nt->z_lexpr;
        if(!std::isnan(nt->ub)){
            std::string str = std::to_string(nt->ub);
            z3::expr b = c.real_val(str.c_str());
            t_expr1 = t_expr1 && nt->nt_z3_var <= b;
        }

        if(!std::isnan(nt->lb)){
            std::string str = std::to_string(nt->lb);
            z3::expr b = c.real_val(str.c_str());
            t_expr1 = t_expr1 && nt->nt_z3_var >= b;
        }

    }
    layer->c_expr = t_expr.simplify();
    layer->b_expr = t_expr1;
    
}

void init_z3_expr_neuron(z3::context &c, Neuron_t* nt){
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

void merged_constraints(z3::context& c, Network_t* net){
    size_t layer_constraint_from=2;
    for(size_t i=layer_constraint_from; i<net->layer_vec.size(); i++){
        Layer_t* curr_layer = net->layer_vec[i];
        if(i==0){
            curr_layer->merged_expr = curr_layer->b_expr && curr_layer->c_expr && net->input_layer->b_expr;
        }
        else if(i == layer_constraint_from){
            curr_layer->merged_expr = curr_layer->b_expr && curr_layer->c_expr && net->layer_vec[i-1]->b_expr;
        }
        else{
            curr_layer->merged_expr = curr_layer->b_expr && curr_layer->c_expr && net->layer_vec[i-1]->merged_expr;
        }
    }

}

void check_sat_output_layer(z3::context& c, Network_t* net){
    z3::solver s(c);
    set_z3_parameters(c);
    s.add(!net->prop_expr && net->layer_vec.back()->b_expr);
    z3::check_result sat_out = s.check();
    z3::model modl = s.get_model();
    if(sat_out == z3::sat){
        sat_var_value(c, modl, net->layer_vec.back());
    }

    for(int i=net->layer_vec.size()-1; i>=0; i--){
        Layer_t* layer = net->layer_vec[i];
        if(layer->is_activation){
            back_prop_relu(c, net, layer);
        }
        else{ // Assuming Affine layer
            //z3::expr t_expr = c.bool_val(true);
            s.reset();
            Layer_t* prev_layer;
            for(auto nt : layer->neurons){
                std::string nt_str = std::to_string(layer->layer_index)+","+std::to_string(nt->neuron_index);
                s.add(nt->affine_expr == nt->nt_z3_var, nt_str.c_str());
                //t_expr = t_expr && (nt->affine_expr == nt->nt_z3_var);
            }
            if(i > 0){
                prev_layer = net->layer_vec[i-1];
            }
            else{
                prev_layer = net->input_layer;
            }
            s.add(prev_layer->b_expr);
            s.add(layer->back_prop_eq);
            //s.add(t_expr && prev_layer->b_expr && layer->back_prop_eq);
            sat_out = s.check();
            if(sat_out == z3::sat){
                modl = s.get_model();
                sat_var_value(c, modl, prev_layer);
                //construct_and_execute_image(i,net);
                std::cout<<"Layer: "<<i<<" is SAT"<<std::endl;
            }
            else{
                std::cout<<"Layer: "<<i<<" is UNSAT"<<std::endl;
                z3::expr_vector core = s.unsat_core();
                std::cout<<"Unsat core: "<<core.size()<<std::endl;
                std::ofstream marked_neuron_file;
                marked_neuron_file.open(Configuration::marked_neuron_path.c_str(), std::ios::out | std::ios::app);
                for(size_t t=0;t<core.size();t++){
                    std::string str = core[t].to_string();
                    std::cout<<"core: "<<str.substr(1,str.size()-2)<<std::endl; 
                    
                    marked_neuron_file << str.substr(1,str.size()-2) <<std::endl;
                }
                break;
            }

        }
    }
}

void back_prop_relu(z3::context& c, Network_t* net, Layer_t* layer){
    z3::expr t_expr = c.bool_val(true);
    Layer_t* prev_layer;
    for(auto nt : layer->neurons){
        if(nt->back_prop_val > 0){
            nt->pred_neurons[0]->back_prop_val = nt->back_prop_val;
            nt->pred_neurons[0]->back_prop_val_expr = nt->back_prop_val_expr;
            t_expr = t_expr && (nt->pred_neurons[0]->nt_z3_var == nt->back_prop_val_expr);
        }
        else if(nt->back_prop_val == 0){
            nt->pred_neurons[0]->back_prop_val = nt->back_prop_val;
            nt->pred_neurons[0]->back_prop_val_expr = nt->back_prop_val_expr;
            z3::expr lb_expr = get_expr_from_double(c,nt->pred_neurons[0]->lb);
            t_expr = t_expr && (lb_expr <= nt->pred_neurons[0]->nt_z3_var) && (nt->pred_neurons[0]->nt_z3_var <= nt->back_prop_val_expr);
        }
    }
    if(layer->layer_index>0){ // Activation layer index will always be greater than 0
        prev_layer = net->layer_vec[layer->layer_index-1];
    }
    else{
        prev_layer = net->input_layer;
    }
    prev_layer->back_prop_eq = t_expr;
    std::cout<<"Layer: "<<layer->layer_index<<" is ReLU"<<std::endl;
}

void sat_var_value(z3::context& c, z3::model &modl, Layer_t* layer){
    z3::expr t_expr = c.bool_val(true);
    for(auto nt : layer->neurons){
        z3::expr val = modl.eval(nt->nt_z3_var);
        nt->back_prop_val_expr = val;
        t_expr = t_expr && (nt->nt_z3_var == val);
        //std::cout<<val<<std::endl;
        nt->back_prop_val = std::stod(val.to_string());
        //std::cout<<nt->nt_z3_var<<" : "<<nt->back_prop_val<<std::endl;
    }
    layer->back_prop_eq = t_expr;
    //std::cout<<layer->back_prop_eq<<std::endl;
}

void affine_expr_init_neuron(z3::context& c, Layer_t* layer, Neuron_t* nt){
    z3::expr sum = get_expr_from_double(c, layer->b(nt->neuron_index));
    for(auto pred_nt:nt->pred_neurons){
        sum = sum + get_expr_from_double(c,layer->w(pred_nt->neuron_index,nt->neuron_index))*pred_nt->nt_z3_var;
    }
    nt->affine_expr = sum.simplify();
}

void affine_expr_init_layer(z3::context& c, Layer_t* layer){
    if(!layer->is_activation){
        //std::cout<<"Layer index: "<<layer->layer_index<<std::endl;
        for(auto nt:layer->neurons){   
            affine_expr_init_neuron(c, layer, nt);
        }
    }
}

void affine_expr_init(z3::context& c, Network_t* net){
    for(auto layer:net->layer_vec){
        affine_expr_init_layer(c,layer);
    }
}

void model_to_image(z3::model &modl, Network_t* net){
    Layer_t* in_layer = net->input_layer;
    std::vector<double> sat_assign;
    z3::set_param("pp.decimal", true);
    for(auto nt : in_layer->neurons){
        std::string val_str = modl.eval(nt->nt_z3_var).to_string();
        double val = std::stod(val_str);
        sat_assign.push_back(val);
        //std::cout<<"Variable: " << nt->nt_z3_var << " = " << modl.eval(nt->nt_z3_var) <<", "<<val<< "\n";
    }
    std::vector<std::size_t> shape = {net->input_dim};
    net->candidate_ce = xt::adapt(sat_assign, shape);

}


