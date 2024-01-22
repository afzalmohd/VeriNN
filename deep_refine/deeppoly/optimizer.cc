#include "optimizer.hh"
#include "helper.hh"
#include "deeppoly_configuration.hh"

pthread_mutex_t lcked;

std::string get_constr_name(size_t layer_idx, size_t nt_idx){
    std::string name = "c_"+std::to_string(layer_idx)+","+std::to_string(nt_idx);
    return name;
}

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

bool is_verified_model_efficiant(GRBModel& model){
    double obj_val;
    for(size_t counter = 1; counter<40; counter++){
        model.set(GRB_DoubleParam_TimeLimit, 5*counter);
        model.optimize();
        int status = model.get(GRB_IntAttr_Status);
        std::cout<<"Verified Opt Status: "<<status<<std::endl;
        if(status == GRB_INFEASIBLE || status == GRB_INF_OR_UNBD){
            return true;
        }
        obj_val = model.get(GRB_DoubleAttr_ObjVal);
        if(obj_val <= 0){
            std::cout<<"Counter: "<<counter<<std::endl;
            std::cout<<"Val: "<<obj_val<<" Bounds: "<<model.get(GRB_DoubleAttr_ObjBound)<<std::endl;
            return false;
        }
        else if(status == GRB_OPTIMAL){
            return true;
        }
        else{
            double bound = model.get(GRB_DoubleAttr_ObjBound);
            if(bound > 0){
                return true;
            }
            std::cout<<"Val: "<<obj_val<<" Bounds: "<<bound<<std::endl;
        }
        if(counter == 5){
            model.set(GRB_IntParam_MIPFocus, GRB_MIPFOCUS_BESTBOUND);
        }
        std::cout<<"Failed Counter: "<<counter<<std::endl;
    }

    
    // model.set(GRB_DoubleParam_TimeLimit, 2000);
    // model.optimize();
    // obj_val = model.get(GRB_DoubleAttr_ObjVal);
    // if(obj_val > 0){
    //     return true;
    // }

    return false;
}

bool is_verified_by_sat_query(GRBModel& model, GRBLinExpr& grb_obj){
    std::cout<<"Checking satisfiability..."<<std::endl;
    std::string out_constr = "removable_constraint";
    model.addConstr(grb_obj, GRB_LESS_EQUAL, 0.0, out_constr);
    GRBLinExpr grb1 = 0;
    model.setObjective(grb1, GRB_MINIMIZE);
    model.set(GRB_DoubleParam_TimeLimit, 2000);
    model.optimize();
    auto rm_constr = model.getConstrByName(out_constr);
    model.remove(rm_constr);
    int status = model.get(GRB_IntAttr_Status);
    if(status == GRB_INFEASIBLE){
        model.update();
        return true;
    }
    else if(status == GRB_OPTIMAL){
        std::cout<<"Optimal result.."<<std::endl;
    }
    else{
        std::cout<<"status: "<<status<<std::endl;
        assert(0 && "Wring grb output\n");
    }

    return false;
}

bool is_verified_model_efficiant_testing(GRBModel& model, std::vector<GRBVar>& var_vector){
    double obj_val;
    for(size_t counter = 1; counter<40; counter++){
        model.set(GRB_DoubleParam_TimeLimit, 5*counter);
        model.optimize();
        int status = model.get(GRB_IntAttr_Status);
        std::cout<<"Verified Opt Status: "<<status<<std::endl;
        obj_val = model.get(GRB_DoubleAttr_ObjVal);
        if(obj_val <= 0){
            size_t var_counter = var_vector.size() - 10;
            for(size_t i=0; i<10; i++){
                std::cout<<"Var idx: "<<i<<" , val: "<<var_vector[var_counter+i].get(GRB_DoubleAttr_X)<<std::endl;
            }
            std::cout<<"Counter: "<<counter<<std::endl;
            std::cout<<"Val: "<<obj_val<<" Bounds: "<<model.get(GRB_DoubleAttr_ObjBound)<<std::endl;
            return false;
        }
        else if(status == GRB_OPTIMAL){
            return true;
        }
        else{
            double bound = model.get(GRB_DoubleAttr_ObjBound);
            if(bound > 0){
                return true;
            }
            std::cout<<"Val: "<<obj_val<<" Bounds: "<<bound<<std::endl;
        }
        if(counter == 5){
            model.set(GRB_IntParam_MIPFocus, GRB_MIPFOCUS_BESTBOUND);
        }
        std::cout<<"Failed Counter: "<<counter<<std::endl;
    }

    
    // model.set(GRB_DoubleParam_TimeLimit, 2000);
    // model.optimize();
    // obj_val = model.get(GRB_DoubleAttr_ObjVal);
    // if(obj_val > 0){
    //     return true;
    // }

    return false;
}

bool verify_by_milp(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector, size_t counter_class_index, bool is_first){
    Layer_t* layer = net->layer_vec.back();
    size_t actual_class_var_index  = get_gurobi_var_index(layer, net->actual_label);
    size_t counter_class_var_index = get_gurobi_var_index(layer, counter_class_index);
    std::string miss_classified_constr = "miss_classified_constr";
    if(Configuration_deeppoly::is_conf_ce){
        model.update();
        model.addConstr(var_vector[counter_class_var_index]-var_vector[actual_class_var_index], GRB_GREATER_EQUAL, 0, miss_classified_constr);
        size_t var_idx = get_gurobi_var_index(layer, 0);
        GRBLinExpr grb_obj = Configuration_deeppoly::conf_of_ce*(var_vector[var_idx]+var_vector[var_idx+1]+var_vector[var_idx+2]+var_vector[var_idx+3]+var_vector[var_idx+4]+var_vector[var_idx+5]+
                                var_vector[var_idx+6]+var_vector[var_idx+7]+var_vector[var_idx+8]+var_vector[var_idx+9]) - var_vector[counter_class_var_index];
        model.setObjective(grb_obj, GRB_MINIMIZE);
        // std::cout<<grb_obj<<std::endl;
    }
    else{
        GRBLinExpr grb_obj = var_vector[actual_class_var_index] - var_vector[counter_class_var_index];
        model.setObjective(grb_obj, GRB_MINIMIZE);
    }
    bool is_verified = false;
    double obj_val;
    is_verified = is_verified_model_efficiant(model);
    if(Configuration_deeppoly::is_conf_ce){
        GRBConstr constr = model.getConstrByName(miss_classified_constr);
        model.remove(constr);
    }
    if(is_verified){
        return true;
    }
    obj_val = model.get(GRB_DoubleAttr_ObjVal);
    // is_verified = is_verified_by_sat_query(model, grb_obj);
    // if(is_verified){
    //     return true;
    // }
    
    // model.set(GRB_DoubleParam_TimeLimit, 2000);
    // model.optimize();
    // obj_val = model.get(GRB_DoubleAttr_ObjVal);
    // if(obj_val > 0){
    //     return true;
    // }
    
    if(is_first){
        std::cout<<"MILP error with ("<<net->actual_label<<","<<counter_class_index<<"): "<<-obj_val<<std::endl;
        update_sat_vals(net, var_vector);
    }
    net->index_vs_err[counter_class_index] = -obj_val;
    // model.update();
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

void create_milp_or_lp_encoding_relu(GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter, Layer_t* layer, size_t nt_index, bool is_with_binary_var){
    std::string contr_name = get_constr_name(layer->layer_index, nt_index);
    std::string constr_name1 = contr_name+",1";
    std::string constr_name2 = contr_name+",2";
    std::string constr_name3 = contr_name+",3";
    std::string constr_name4 = contr_name+",4";
    Neuron_t* pred_nt = layer->pred_layer->neurons[nt_index];
    double lb = -pred_nt->lb;
    std::string var_str = "bin,"+std::to_string(layer->layer_index)+","+std::to_string(nt_index);
    GRBVar var;
    if(is_with_binary_var){
        var = model.addVar(0,1,0,GRB_BINARY, var_str);
    }
    else{
        var = model.addVar(0,1,0,GRB_CONTINUOUS, var_str);
    }
    
    GRBLinExpr grb_expr = var_vector[var_counter+nt_index] - var_vector[var_counter + nt_index - layer->pred_layer->dims] - lb*var;
    model.addConstr(grb_expr, GRB_LESS_EQUAL, -lb, constr_name1);

    grb_expr = var_vector[var_counter+nt_index] - var_vector[var_counter + nt_index - layer->pred_layer->dims];
    model.addConstr(grb_expr, GRB_GREATER_EQUAL, 0, constr_name2);

    grb_expr = var_vector[var_counter+nt_index] - pred_nt->ub*var;
    model.addConstr(grb_expr, GRB_LESS_EQUAL, 0, constr_name3);

    grb_expr = var_vector[var_counter+nt_index];
    model.addConstr(grb_expr, GRB_GREATER_EQUAL, 0, constr_name4);
}

void create_deeppoly_encoding_relu(GRBModel& model, Layer_t* layer, size_t nt_index, std::vector<GRBVar>& var_vector, size_t var_counter){
    std::string contr_name = get_constr_name(layer->layer_index, nt_index);
    std::string constr_name1 = contr_name+",1";
    std::string constr_name2 = contr_name+",2";
    Neuron_t* nt = layer->neurons[nt_index];
    Neuron_t* pred_nt = layer->pred_layer->neurons[nt_index];
    double lb = -pred_nt->lb;
    GRBLinExpr grb_expr = -pred_nt->ub*var_vector[var_counter + nt_index - layer->pred_layer->dims] + pred_nt->ub*lb;
    grb_expr += (nt->ub - lb)*var_vector[var_counter+nt_index];
    model.addConstr(grb_expr, GRB_LESS_EQUAL, 0, constr_name1);

    GRBLinExpr grb_expr1 = var_vector[var_counter+nt_index] - var_vector[var_counter + nt_index - layer->pred_layer->dims];
    model.addConstr(grb_expr1, GRB_GREATER_EQUAL, 0, constr_name2);
}

void create_milp_constr_relu(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter){
    assert(layer->is_activation && "Not activation layer\n");
    for(size_t i=0; i< layer->dims; i++){
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
                if(IS_LP_CONSTRAINTS){
                    create_milp_or_lp_encoding_relu(model, var_vector, var_counter, layer, i, false);
                }
                else{
                    create_deeppoly_encoding_relu(model, layer, i, var_vector, var_counter);
                }               
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
    model.set(GRB_IntParam_Threads,NUM_GUROBI_THREAD);
    return model;
}

void update_sat_vals(Network_t* net, std::vector<GRBVar>& var_vec){
    for(Neuron_t* nt : net->input_layer->neurons){
        nt->sat_val = var_vec[nt->neuron_index].get(GRB_DoubleAttr_X);
    }
    size_t counter = net->input_layer->dims;
    size_t layer_index;
    for(layer_index=0; layer_index<net->layer_vec.size()-1; layer_index++){
        Layer_t* layer = net->layer_vec[layer_index];
        for(Neuron_t* nt : layer->neurons){
            nt->sat_val = var_vec[counter+nt->neuron_index].get(GRB_DoubleAttr_X);
        }
        counter += net->layer_vec[layer_index]->dims;
    }
    Layer_t* last_layer = net->layer_vec[layer_index];
    for(Neuron_t* nt : last_layer->neurons){
        nt->sat_val = var_vec[counter+nt->neuron_index].get(GRB_DoubleAttr_X);
        // std::cout<<"Var idx: "<<nt->neuron_index<<" , val: "<<nt->sat_val<<std::endl;
    }
}