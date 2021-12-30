#include "pullback.hh"


bool pull_back_full(Network_t* net){
    for(int i = net->layer_vec.size()-1; i>=0; i--){
        Layer_t* layer = net->layer_vec[i];
        if(layer->is_activation){
            pull_back_relu(layer);
        }
        else{
            bool is_infisible = pull_back_FC(layer);
            if(is_infisible){
                if(layer->is_marked){
                    std::cout<<"Marked layer: "<<layer->layer_index<<", Marked neurons: ";
                    for(auto nt: layer->neurons){
                        if(nt->is_marked){
                            std::cout<<nt->neuron_index<<" ";
                        }
                    }
                    std::cout<<std::endl;
                }
                return false;
            }
            else{
                if(layer->layer_index == 0){
                    return true; //Found counter example
                }
            }
        }
    }
    return false;
}

void pull_back_relu(Layer_t* layer){
    assert(layer->is_activation && "Layer does not have activation function\n");
    Layer_t* pred_layer = layer->pred_layer;
    for(size_t i=0; i<layer->dims; i++){
        Neuron_t* nt = layer->neurons[i];
        if(nt->is_back_prop_active){
            Neuron_t* pred_nt = pred_layer->neurons[i];
            assert(nt->back_prop_lb == nt->back_prop_ub && "Relu output node contain two values during pull back\n");
            if(nt->back_prop_lb > 0){
                pred_nt->back_prop_lb = nt->back_prop_lb;
                pred_nt->back_prop_ub = nt->back_prop_lb;
            }
            else{
                pred_nt->back_prop_ub = 0;
                pred_nt->back_prop_lb = -pred_nt->lb;
            }
            pred_nt->is_back_prop_active = true;
        }
    }
}

bool pull_back_FC(Layer_t* layer){
    assert(layer->layer_type == "FC" && "Layer is not FC\n");
    Layer_t* pred_layer = layer->pred_layer;
    GRBEnv env = GRBEnv(true);
    env.start();
    GRBModel model = GRBModel(env);
    std::vector<GRBVar> var_vector;
    var_vector.reserve(pred_layer->dims);
    create_gurobi_variable(model, var_vector, pred_layer);
    create_layer_constrains_pullback(model, var_vector, layer);
    model.optimize();
    int status = model.get(GRB_IntAttr_Status);
    if(status == GRB_OPTIMAL){
        for(size_t i=0; i<var_vector.size(); i++){
            Neuron_t* nt = pred_layer->neurons[i];
            nt->back_prop_ub = var_vector[i].get(GRB_DoubleAttr_X);
            nt->back_prop_lb = nt->back_prop_ub;
            nt->is_back_prop_active = true;
        }
        if(pred_layer->layer_index == -1){
            std::cout<<"Counter example found!!"<<std::endl;
        }
        return false;
    }
    else if(status == GRB_INFEASIBLE){
        model.computeIIS();
        std::cout<<"Infisible constrains: "<<std::endl;
        GRBConstr* c = model.getConstrs();
        bool is_iss = false;
        for(int i=0; i<model.get(GRB_IntAttr_NumConstrs); i++){
            if(c[i].get(GRB_IntAttr_IISConstr) == 1){
                std::string name = c[i].get(GRB_StringAttr_ConstrName);
                std::string index_str(name.substr(1));
                size_t nt_index = std::stoul(index_str);
                layer->neurons[nt_index]->is_marked = true;
                is_iss = true;
            }
        }
        layer->is_marked = true;
        return true;
    }
    return false;
}

void create_gurobi_variable(GRBModel& model, std::vector<GRBVar>& var_vector, Layer_t* layer){
    for(auto nt:layer->neurons){
        std::string var_str = "x"+std::to_string(nt->neuron_index);
        GRBVar var = model.addVar(-nt->lb, nt->ub, 0.0, GRB_CONTINUOUS, var_str);
        var_vector.push_back(var);
    }
}

void create_layer_constrains_pullback(GRBModel& model, std::vector<GRBVar>& var_vector, Layer_t* layer){
    for(auto nt : layer->neurons){
        if(nt->is_back_prop_active){
            std::string constr_str = "c"+std::to_string(nt->neuron_index);
            GRBLinExpr grb_expr;
            grb_expr.addTerms(&nt->uexpr->coeff_sup[0], &var_vector[0], var_vector.size());
            grb_expr += nt->uexpr->cst_sup;
            if(nt->back_prop_lb == nt->back_prop_ub){
                model.addConstr(grb_expr == nt->back_prop_ub, constr_str);
            }
            else{
                model.addConstr(grb_expr >= nt->back_prop_lb, constr_str);
                model.addConstr(grb_expr <= nt->back_prop_ub, constr_str);
            }
        }
    }
}