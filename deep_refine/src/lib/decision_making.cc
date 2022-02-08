#include "decision_making.hh"
#include "pullback.hh"
#include "../../deeppoly/helper.hh"

bool marked_neurons_vector(Network_t* net, std::vector<std::vector<Neuron_t*>>& marked_nt){
    std::vector<Neuron_t*> nt_vec;
    for(auto layer : net->layer_vec){
        if(layer->is_marked){
            for(auto nt : layer->neurons){
                if(nt->is_marked && !is_duplicate_neuron_marked(net, nt)){
                    nt_vec.push_back(nt);
                    layer->marked_neurons.push_back(nt);
                }
            }
        }
    }
    if(nt_vec.size() > 0){
        marked_nt.push_back(nt_vec);
        return true; // New neurons added
    }
    return false;// No new neuron added
}

bool is_duplicate_neuron_marked(Network_t* net, Neuron_t* nt){
    assert(nt->layer_index >= 0 && "Layer index is out of range\n");
    Layer_t* layer = net->layer_vec[nt->layer_index];
    for(auto nt1 : layer->marked_neurons){
        if(nt1->neuron_index == nt->neuron_index){
            return true;
        }
    }
    return false;
}

bool set_marked_path(Network_t* net, std::vector<std::vector<Neuron_t*>>& marked_vec, bool is_first){
    std::vector<Neuron_t*> nt_vec = marked_vec.back();
    bool is_pred;
    if(is_first){
        is_pred = set_to_predecessor(nt_vec, is_first);
        if(!is_pred){
            return false;
        }
        return set_pred_path_if_not_valid(net, marked_vec);
    }
    else{
        is_pred = set_to_predecessor(nt_vec, false);
        while(!is_pred){
            reset_mark__nt(net, nt_vec);
            nt_vec.clear();
            marked_vec.pop_back();
            if(marked_vec.empty()){
                return false;
            }
            else{
                nt_vec = marked_vec.back();
                is_pred = set_to_predecessor(nt_vec, false);
            }
        }
        return set_pred_path_if_not_valid(net, marked_vec);
    }
}

bool set_pred_path_if_not_valid(Network_t* net, std::vector<std::vector<Neuron_t*>>& marked_vec){
    std::vector<Neuron_t*> nt_vec = marked_vec.back();
    bool is_pred;
    while(!is_valid_path(net, nt_vec)){
        is_pred = set_to_predecessor(nt_vec, false);
        while(!is_pred){
            reset_mark__nt(net, nt_vec);
            nt_vec.clear();
            marked_vec.pop_back();
            if(marked_vec.empty()){
                return false;
            }
            else{
                nt_vec = marked_vec.back();
                is_pred = set_to_predecessor(nt_vec, false);
            }
        }
    }
    return true;
}

bool is_valid_path(Network_t* net, std::vector<Neuron_t*>& nt_vec){
    std::vector<bool> is_visited_layer(net->layer_vec.size(), false);
    for(Neuron_t* nt : nt_vec){
        Layer_t* layer = net->layer_vec[nt->layer_index];
        if(!is_visited_layer[nt->layer_index]){
            is_visited_layer[nt->layer_index] = true;
            if(is_valid_path_with_iss(layer)){
                if(!is_valid_path_with_milp(layer)){ 
                    return false;
                }
            }
            else{
                return false;
            }

        }
    }

    return true;
}


void reset_mark__nt(Network_t* net, std::vector<Neuron_t*>& nt_vec){
    for(Neuron_t* nt : nt_vec){
        Layer_t* layer = net->layer_vec[nt->layer_index];
        nt->is_marked = false;
        std::vector<Neuron_t*>::iterator itr = std::remove(layer->marked_neurons.begin(), layer->marked_neurons.end(), nt);
        layer->marked_neurons.erase(itr, layer->marked_neurons.end());
    }
}

bool set_to_predecessor(std::vector<Neuron_t*>& nt_vector, bool is_first){
    if(is_first){
        for(auto nt : nt_vector){
            nt->is_active = true;
        }
        return true;
    }  

    int i = nt_vector.size() - 1;
    if(i < 0){
        return false;
    }
    
    while(!nt_vector[i]->is_active){
        i--;
        if(i < 0){
            return false;
        }
    }
    if(i < 0){
        return false;
    }

    nt_vector[i]->is_active = false;
    for(size_t j = i+1; j < nt_vector.size(); j++){
        nt_vector[j]->is_active = true;
    }

    return true;
}

bool is_valid_path_with_iss(Layer_t* layer){
    bool is_valid = false;
    for(size_t i=0; i < layer->IIS.size(); i++){
        is_valid = false;
        std::vector<Sparse_neuron_t*> one_iis = layer->IIS[i];
        for(auto snt : one_iis){
            Neuron_t* nt = layer->neurons[snt->neuron_index];
            if(nt->is_active != snt->is_active){
                is_valid =true;
                break;
            }
        }
        if(!is_valid){
            return false;
        }
    }
    return true;
}

bool is_valid_path_with_milp(Layer_t* layer){
    assert(layer->layer_type == "FC" && "Layer is not FC\n");
    Layer_t* pred_layer = layer->pred_layer;
    GRBModel model = create_grb_env_and_model();
    std::vector<GRBVar> var_vector;
    var_vector.reserve(pred_layer->dims);
    create_gurobi_variable_with_unmarked_bounds(model, var_vector, pred_layer);
    create_layer_constrains_for_valid_path(model, var_vector, layer);
    model.optimize();
    int status = model.get(GRB_IntAttr_Status);
    if(status == GRB_OPTIMAL){
        return true;
    }
    else if(status == GRB_INFEASIBLE){
        model.computeIIS();
        std::vector<Sparse_neuron_t*> snt_vec;
        std::cout<<"Infisible constrains in valid path: "<<std::endl;
        GRBConstr* c = model.getConstrs();
        for(int i=0; i<model.get(GRB_IntAttr_NumConstrs); i++){
            if(c[i].get(GRB_IntAttr_IISConstr) == 1){
                std::string name = c[i].get(GRB_StringAttr_ConstrName);
                std::string index_str(name.substr(1));
                size_t nt_index = std::stoul(index_str);
                Neuron_t* nt = layer->marked_neurons[nt_index];
                Sparse_neuron_t* snt = new Sparse_neuron_t();
                snt->is_active = nt->is_active;
                snt->layer_index = nt->layer_index;
                snt->neuron_index = nt->neuron_index;
                snt_vec.push_back(snt);
            }
        }
        if(!snt_vec.empty()){
            layer->IIS.push_back(snt_vec);
        }
    }
    return false;
}

void create_layer_constrains_for_valid_path(GRBModel& model, std::vector<GRBVar>& var_vector, Layer_t* layer){
    for(Neuron_t* nt : layer->marked_neurons){
        std::vector<double> coeffs = get_neuron_incomming_weigts(nt, layer);
        double cst = get_neuron_bias(nt, layer);
        std::string constr_str = "c"+std::to_string(nt->neuron_index);
        GRBLinExpr grb_expr;
        grb_expr.addTerms(&coeffs[0], &var_vector[0], var_vector.size());
        grb_expr += cst;
        if(nt->is_active){
            model.addConstr(grb_expr, GRB_GREATER_EQUAL, 0, constr_str);
        }
        else{
            model.addConstr(grb_expr, GRB_LESS_EQUAL, 0, constr_str);
        }
    }
}

void create_gurobi_variable_with_unmarked_bounds(GRBModel& model, std::vector<GRBVar>& var_vector, Layer_t* layer){
    for(auto nt:layer->neurons){
        std::string var_str = "x"+std::to_string(nt->neuron_index);
        GRBVar var = model.addVar(-nt->unmarked_lb, nt->unmarked_ub, 0.0, GRB_CONTINUOUS, var_str);
        var_vector.push_back(var);
    }
}

