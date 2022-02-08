#include "optimizer.hh"
#include "helper.hh"

// void compute_bounds_using_gurobi(Network_t* net, Layer_t* layer, Neuron_t* nt, Expr_t* expr, bool is_minimize){
//     Layer_t* pred_layer = get_pred_layer(net, layer);
//     try{
//         GRBModel model = create_env_and_model();
//         std::vector<GRBVar> var_vector(pred_layer->dims);
//         //add variables to the model
//         for(size_t i=0; i<pred_layer->dims; i++){
//             Neuron_t* nt1 = pred_layer->neurons[i];
//             std::string var_str = "i_"+std::to_string(i);
//             GRBVar x = model.addVar(-nt1->lb, nt1->ub, 0.0, GRB_CONTINUOUS, var_str);
//             var_vector[i] = x;
//         }

//         //add objective function to the model
//         GRBLinExpr obj_expr = 0;
//         if(is_minimize){
//             std::vector<double> coeffs;
//             copy_vector_with_negative_vals(expr->coeff_inf, coeffs);
//             obj_expr.addTerms(&coeffs[0], &var_vector[0], var_vector.size());
//             obj_expr -= expr->cst_inf;
//         }
//         else{
//             obj_expr.addTerms(&expr->coeff_sup[0], &var_vector[0], var_vector.size());
//             obj_expr += expr->cst_sup;
//         }
        
//         if(is_minimize){
//             model.setObjective(obj_expr, GRB_MINIMIZE);
//         }
//         else{
//             model.setObjective(obj_expr, GRB_MAXIMIZE);
//         }

//         for(auto con : expr->constr_vec){
//             Expr_t* con_expr = con->expr;
//             GRBLinExpr grb_expr = 0;
//             if(con->is_positive){
//                 grb_expr.addTerms(&con_expr->coeff_sup[0], &var_vector[0], var_vector.size());
//                 grb_expr += con_expr->cst_sup;
//                 model.addConstr(grb_expr, GRB_GREATER_EQUAL, 0);
//             }
//             else{
//                 std::vector<double> coeffs;
//                 copy_vector_with_negative_vals(expr->coeff_inf, coeffs);
//                 grb_expr.addTerms(&coeffs[0], &var_vector[0], var_vector.size());
//                 grb_expr -= con_expr->cst_inf; //cst_inf already in negative form
//                 model.addConstr(grb_expr, GRB_LESS_EQUAL, 0); 
//             }
//         }

//         model.optimize();
//         int status = model.get(GRB_IntAttr_Status);
//         if(status == GRB_OPTIMAL){
//             if(is_minimize){
//             double tmp = model.get(GRB_DoubleAttr_ObjVal);
//             if(tmp > -nt->lb){
//                 nt->lb = -tmp;
//             }
//             }
//             else{
//                 double tmp = model.get(GRB_DoubleAttr_ObjVal);
//                 if(tmp < nt->ub){
//                     nt->ub = tmp;
//                 }
//             }
//         }
//         else if(status == GRB_INFEASIBLE){
//             std::cout<<"Infisible bounds"<<std::endl;
//         }
//         else if(status == GRB_UNBOUNDED){
//             std::cout<<"UNBOUNDED bounds"<<std::endl;
//         }
//         else{
//             std::cout<<"UNKNOWN bounds"<<std::endl;
//         }
        

//     }
//     catch(GRBException e){
//         std::cout<<e.getMessage()<<std::endl;
//     }
//     catch(...){
//         std::cout<<"Exception during optimization"<<std::endl;
//     }

// }


GRBModel create_env_model_constr(Network_t* net, std::vector<GRBVar>& var_vector){
    GRBModel model = create_env_and_model();
    creating_variables(net, model, var_vector);
    size_t var_counter = net->input_layer->dims;
    for(auto layer : net->layer_vec){
        if(layer->is_activation){
            create_milp_constr_relu(layer, model, var_vector, var_counter);
        }
        else{
            create_milp_constr_FC(layer, model, var_vector, var_counter);
        }
        var_counter += layer->dims;
    }
    return model;
}

bool verify_by_milp(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector, size_t counter_class_index, bool is_first){
    Layer_t* layer = net->layer_vec.back();
    size_t actual_class_var_index  = get_gurobi_var_index(layer, net->actual_label);
    size_t counter_class_var_index = get_gurobi_var_index(layer, counter_class_index);
    GRBLinExpr grb_obj = var_vector[actual_class_var_index] - var_vector[counter_class_var_index];
    model.setObjective(grb_obj, GRB_MINIMIZE);
    model.optimize();
    double obj_val = model.get(GRB_DoubleAttr_ObjVal);
    if(obj_val > 0){
        return true;
    }
    // std::cout<<var_vector[actual_class_var_index].get(GRB_StringAttr_VarName)<<" "<<var_vector[actual_class_var_index].get(GRB_DoubleAttr_X)<<std::endl;
    // std::cout<<var_vector[counter_class_var_index].get(GRB_StringAttr_VarName)<<" "<<var_vector[counter_class_var_index].get(GRB_DoubleAttr_X)<<std::endl;
    if(is_first){
        std::cout<<"MILP error with ("<<net->actual_label<<","<<counter_class_index<<"): "<<-obj_val<<std::endl;
        Neuron_t* nt_actual = layer->neurons[net->actual_label];
        Neuron_t* nt_counter = layer->neurons[counter_class_index];
        nt_actual->is_back_prop_active = true;
        nt_actual->back_prop_lb = var_vector[actual_class_var_index].get(GRB_DoubleAttr_X);
        nt_actual->back_prop_ub = nt_actual->back_prop_lb;
        nt_counter->is_back_prop_active = true;
        nt_counter->back_prop_lb = var_vector[counter_class_var_index].get(GRB_DoubleAttr_X);
        nt_counter->back_prop_ub = nt_counter->back_prop_lb;
        update_sat_vals(net, var_vector);
    }
    return false;
} 

void creating_variables_one_layer(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector, Layer_t* layer){
    if(layer->is_activation){
        for(auto nt: layer->neurons){
            std::string var_str = "x,"+std::to_string(layer->layer_index)+","+std::to_string(nt->neuron_index);
            GRBVar x = model.addVar(-nt->lb, nt->ub, 0.0, GRB_CONTINUOUS, var_str);
            var_vector.push_back(x);
        }
    }
    else{
        for(auto nt : layer->neurons){
            std::string var_str = "x,"+std::to_string(layer->layer_index)+","+std::to_string(nt->neuron_index);
            GRBVar x;
            if(nt->is_marked){
                if(nt->is_active){
                    x = model.addVar(0, nt->ub, 0.0, GRB_CONTINUOUS, var_str);
                }
                else{
                    x = model.addVar(-nt->lb, 0, 0.0, GRB_CONTINUOUS, var_str);
                }
            }
            else{
                x = model.addVar(-nt->lb, nt->ub, 0.0, GRB_CONTINUOUS, var_str);
            }   
            var_vector.push_back(x);
        } 
    }
}

void creating_variables(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector){
    creating_variables_one_layer(net, model, var_vector, net->input_layer);
    for(auto layer : net->layer_vec){
        creating_variables_one_layer(net, model, var_vector, layer);
    }
}

void create_milp_constr_FC_node(Neuron_t* nt, GRBModel& model, std::vector<GRBVar>& var_vector, std::vector<GRBVar>& new_vec, size_t var_counter){
    GRBLinExpr grb_expr;// = -1*var_vector[end_index+i];
    grb_expr.addTerms(&nt->uexpr->coeff_sup[0], &new_vec[0], new_vec.size());
    grb_expr += nt->uexpr->cst_sup;
    grb_expr += -1*var_vector[var_counter+nt->neuron_index];
    model.addConstr(grb_expr, GRB_EQUAL, 0);
    if(nt->is_marked){
        if(nt->is_active){
            model.addConstr(var_vector[var_counter+nt->neuron_index], GRB_GREATER_EQUAL, 0);
        }
        else{
            model.addConstr(var_vector[var_counter+nt->neuron_index], GRB_LESS_EQUAL, 0);
        }
    }
}

void create_milp_constr_FC(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter){    
    assert(layer->layer_type == "FC" && "Layer type is not FC");
    size_t start_index = var_counter - layer->pred_layer->dims;
    size_t end_index = var_counter;
    std::vector<GRBVar> new_vec;
    new_vec.reserve(layer->pred_layer->dims);
    copy_vector_by_index(var_vector, new_vec, start_index, end_index);
    for(size_t i=0; i<layer->dims; i++){
        Neuron_t* nt = layer->neurons[i];
        create_milp_constr_FC_node(nt, model, var_vector, new_vec, var_counter);
    }
}

void create_milp_constr_relu(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter){
    assert(layer->is_activation && "Not activation layer\n");
    for(size_t i=0; i< layer->dims; i++){
        Neuron_t* nt = layer->neurons[i];
        Neuron_t* pred_nt = layer->pred_layer->neurons[i];
        if(pred_nt->is_marked){
            if(pred_nt->is_active){
                GRBLinExpr grb_expr = var_vector[var_counter+i] - var_vector[var_counter + i - layer->pred_layer->dims];
                model.addConstr(grb_expr, GRB_EQUAL, 0);
            }
            else{
                GRBLinExpr grb_expr = var_vector[var_counter + i];
                model.addConstr(grb_expr, GRB_EQUAL, 0);
            }
        }
        else{
            if(pred_nt->lb <= 0){
                GRBLinExpr grb_expr = var_vector[var_counter+i] - var_vector[var_counter + i - layer->pred_layer->dims];
                model.addConstr(grb_expr, GRB_EQUAL, 0);
            }
            else if(pred_nt->ub <= 0){
                GRBLinExpr grb_expr = var_vector[var_counter + i];
                model.addConstr(grb_expr, GRB_EQUAL, 0);
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
}

void copy_vector_by_index(std::vector<GRBVar>& var_vector, std::vector<GRBVar>& new_vec, size_t start_index, size_t end_index){
    for(size_t i=start_index; i < end_index; i++){
        new_vec.push_back(var_vector[i]);
    }
}

size_t get_gurobi_var_index(Layer_t* layer, size_t index){
    Layer_t* pred_layer = layer->pred_layer;
    size_t count=0;
    while(pred_layer->layer_index >= 0){
        count += pred_layer->dims;
        pred_layer = pred_layer->pred_layer;
    }
    if(pred_layer->layer_index == -1){
        count += pred_layer->dims;
    }
    count += index;
    return count;
}

GRBModel create_env_and_model(){
    GRBEnv env = GRBEnv(true);
    env.start();
    GRBModel model = GRBModel(env); 
    model.set(GRB_IntParam_LogToConsole, 0);
    return model;
}

void update_sat_vals(Network_t* net, std::vector<GRBVar>& var_vec){
    for(Neuron_t* nt : net->input_layer->neurons){
        nt->sat_val = var_vec[nt->neuron_index].get(GRB_DoubleAttr_X);
    }
    size_t counter = net->input_layer->dims;
    size_t layer_index;
    for(layer_index=0; layer_index<net->layer_vec.size()-1; layer_index++){
        counter += net->layer_vec[layer_index]->dims;
    }
    Layer_t* last_layer = net->layer_vec[layer_index];
    for(Neuron_t* nt : last_layer->neurons){
        nt->sat_val = var_vec[counter+nt->neuron_index].get(GRB_DoubleAttr_X);
    }
}