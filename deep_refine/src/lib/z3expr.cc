#include "z3expr.hh" //network.hh included in z3expr.hh
#include "backprop.hh"
#include <ctime>
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
    // for(size_t i=net->layer_vec.size(); i>=0 ; i--){

    // }
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
    z3::set_param("pp.decimal-precision", 5);
    //s.add(net->input_layer->b_expr);
    s.add(!net->prop_expr);
    s.add(net->layer_vec.back()->merged_expr);
    //std::cout << s;
    auto sat_out = s.check();
    std::cout<<sat_out<<std::endl;
    if(sat_out == z3::sat){
        z3::model m = s.get_model();
        model_to_image(m, net);
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


