#include "milp_refine.hh"
#include "pullback.hh"
#include "../../deeppoly/analysis.hh"
#include "../../deeppoly/optimizer.hh"

bool is_image_verified_by_milp(Network_t* net){
    reset_backprop_vals(net);
    GRBModel model = create_grb_env_and_model();
    std::vector<GRBVar> var_vector;
    creating_vars_milp(net, model, var_vector);
    size_t var_counter = net->input_layer->dims;
    for(auto layer : net->layer_vec){
        if(layer->is_activation){
            create_relu_constr_milp_refine(layer, model, var_vector, var_counter);
        }
        else{
            create_milp_constr_FC(layer, model, var_vector, var_counter);
        }
        var_counter += layer->dims;
    }

    for(size_t i=0; i<net->output_dim; i++){
        if(i != net->actual_label){
            if(!is_greater(net, net->actual_label, i)){
                if(!verify_by_milp(net, model, var_vector, i)){
                    return false;
                }
            }
        }
    }

    return true;
}

void creating_vars_milp(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector){
    create_vars_layer(net->input_layer, model, var_vector);
    for(auto layer : net->layer_vec){
        create_vars_layer(layer, model, var_vector);
    }
}

void create_vars_layer(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector){
    for(auto nt: layer->neurons){
        std::string var_str = "x,"+std::to_string(layer->layer_index)+","+std::to_string(nt->neuron_index);
        GRBVar x = model.addVar(-nt->lb, nt->ub, 0.0, GRB_CONTINUOUS, var_str);
        var_vector.push_back(x);
    }
}

void create_relu_constr_milp_refine(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter){
    assert(layer->is_activation && "Not activation layer\n");
    for(size_t i=0; i< layer->dims; i++){
        Neuron_t* nt = layer->neurons[i];
        Neuron_t* pred_nt = layer->pred_layer->neurons[i];
        if(pred_nt->lb <= 0){
            GRBLinExpr grb_expr = var_vector[var_counter+i] - var_vector[var_counter + i - layer->pred_layer->dims];
            model.addConstr(grb_expr, GRB_EQUAL, 0);
        }
        else if(pred_nt->ub <= 0){
            GRBLinExpr grb_expr = var_vector[var_counter + i];
            model.addConstr(grb_expr, GRB_EQUAL, 0);
        }
        else if(pred_nt->is_marked){
            double lb = -pred_nt->lb;
            std::string var_str = "bin,"+std::to_string(layer->layer_index)+","+std::to_string(nt->neuron_index);
            GRBVar bin_var = model.addVar(0,1,0,GRB_BINARY, var_str);
            
            GRBLinExpr grb_expr = var_vector[var_counter+i] - var_vector[var_counter + i - layer->pred_layer->dims] - lb*bin_var;
            model.addConstr(grb_expr, GRB_LESS_EQUAL, -lb);

            grb_expr = var_vector[var_counter+i] - var_vector[var_counter + i - layer->pred_layer->dims];
            model.addConstr(grb_expr, GRB_GREATER_EQUAL, 0);

            grb_expr = var_vector[var_counter+i] - pred_nt->ub*bin_var;
            model.addConstr(grb_expr, GRB_LESS_EQUAL, 0);

            grb_expr = var_vector[var_counter+i];
            model.addConstr(grb_expr, GRB_GREATER_EQUAL, 0);

            model.addGenConstrIndicator(bin_var, true, var_vector[var_counter + i - layer->pred_layer->dims], GRB_GREATER_EQUAL, 0.0);
            model.addGenConstrIndicator(bin_var, false, var_vector[var_counter + i - layer->pred_layer->dims], GRB_LESS_EQUAL, 0.0);
        }
        else{
            double lb = -pred_nt->lb;
            GRBLinExpr grb_expr = -pred_nt->ub*var_vector[var_counter + i - layer->pred_layer->dims] + pred_nt->ub*lb;
            grb_expr += (nt->ub - lb)*var_vector[var_counter+i];
            model.addConstr(grb_expr, GRB_LESS_EQUAL, 0);

            GRBLinExpr grb_expr1 = var_vector[var_counter+i] - var_vector[var_counter + i - layer->pred_layer->dims];
            model.addConstr(grb_expr1, GRB_GREATER_EQUAL, 0);
        }

    }
}

