#include "milp_refine.hh"
#include "../../deeppoly/analysis.hh"
#include "../../deeppoly/optimizer.hh"
#include "../../deeppoly/deeppoly_configuration.hh"
#include "milp_mark.hh"

void unmark_net(Network_t* net){
    for(Layer_t* layer : net->layer_vec){
        if(!layer->is_activation && layer->is_marked){
            layer->is_marked = false;
            for(Neuron_t* nt : layer->neurons){
                if(nt->is_marked){
                    nt->is_marked = false;
                    Global_vars::num_marked_neurons += 1;
                }
            }
        }
    }
}

void testing(Network_t* net){
    GRBModel model = create_grb_env_and_model();
    std::vector<GRBVar> var_vector;
    create_milp_mark_milp_refine_constr(net, model, var_vector);

    size_t var_counter = 0;
    update_vars_bounds_by_prev_satval(net->input_layer, var_vector, var_counter);
    var_counter = net->input_dim;
    update_vars_bounds_by_prev_satval(net->layer_vec[0], var_vector, var_counter);
    var_counter = var_vector.size() - net->output_dim;
    update_vars_bounds_by_prev_satval(net->layer_vec.back(), var_vector, var_counter);

    Layer_t* layer = net->layer_vec[1];
    auto iter = layer->res.begin();
    for(size_t i=0; i<layer->dims; i++, iter++){
        std::cout<<"Vals: "<<layer->res[i]<<" , "<<*iter<<std::endl;
    }
    

    model.optimize();
    int status = model.get(GRB_IntAttr_Status);

    std::cout<<"Testing model status: "<<status<<std::endl;

}

void get_marked_neurons(GRBModel& model,  Network_t* net, std::vector<GRBVar>& var_vector){
    bool is_already_optimized = false;
    
    size_t var_counter = 0;
    update_vars_bounds_by_prev_satval(net->input_layer, var_vector, var_counter);
    var_counter = net->input_dim;
    update_vars_bounds_by_prev_satval(net->layer_vec[0], var_vector, var_counter);
    var_counter = var_vector.size() - net->output_dim;
    update_vars_bounds_by_prev_satval(net->layer_vec.back(), var_vector, var_counter);
    // std::string file_name = "/home/u1411251/Documents/tools/conf_ce/VeriNN/deep_refine/grb_constr";
    var_counter = net->input_dim;
    for(Layer_t* layer : net->layer_vec){
        if(layer->is_activation){
            create_optimization_constraints_layer(layer, model, var_vector, var_counter);
            // file_name = file_name+std::to_string(net->number_of_refine_iter)+".mps";
            std::cout<<"Refine iter number: "<<Global_vars::iter_counts<<std::endl;
            // model.write(file_name);
            model.optimize();
            int status = model.get(GRB_IntAttr_Status);
            std::cout<<"Layer index: "<<layer->layer_index<<std::endl;
            std::cout<<"Optimized status: "<<status<<std::endl;
            bool is_layer_marked = false;
            // is_layer_marked = is_layer_marked_after_optimization(layer, var_vector, var_counter);
            if(status = GRB_OPTIMAL){
                is_layer_marked = is_layer_marked_after_optimization(layer, var_vector, var_counter);
            }
            else{
                is_layer_marked = is_layer_marked_after_optimization_without_maxsat(layer);
            }
            
            is_already_optimized = true;
            if(is_layer_marked){
                break;
            }
            else{
                update_vars_bounds(layer, var_vector, var_counter);
            }
        }
        else if(is_already_optimized){
            // std::cout<<"Layer index: "<<layer->layer_index<<std::endl;
            update_vars_bounds(layer, var_vector, var_counter);
            remove_maxsat_constr(model, layer->pred_layer);
            model.update();
            // std::cout<<"Layer index: "<<layer->layer_index<<std::endl;
        }
        var_counter += layer->dims;
    }
}

bool is_layer_marked_after_optimization_without_maxsat(Layer_t* start_layer){
    bool is_marked = false;
    std::map<Neuron_t*, double> nt_err_map;
    auto res_iter = start_layer->res.begin();
    for(size_t i=0; i<start_layer->neurons.size(); i++, res_iter++){
        Neuron_t* pred_nt = start_layer->pred_layer->neurons[i];
        Neuron_t* nt = start_layer->neurons[i];
        // GRBVar var = var_vector[var_counter+i];
        // std::cout<<"Marking......."<<std::endl;
        // double sat_val = var.get(GRB_DoubleAttr_X);
        double res = *res_iter;
        double diff = abs(nt->sat_val - res);
        // std::cout<<diff<<" , "<<sat_val<<" , "<<res<<std::endl;
        if(diff > DIFF_TOLERANCE){
            if(pred_nt->lb > 0 && pred_nt->ub > 0){
                is_marked = true;
                nt_err_map[pred_nt] = diff;
            }
        }
    }
    // std::cout<<std::endl;
    std::cout<<"Layer index: "<<start_layer->pred_layer->layer_index<<", marked neurons: ";
    if(nt_err_map.size() > MAX_NUM_MARKED_NEURONS){
        for(size_t i = 0; i<MAX_NUM_MARKED_NEURONS; i++){
            if(nt_err_map.size() > 0){
                Neuron_t* max_val_nt = NULL;
                if(IS_TOP_MIN_DIFF){
                    max_val_nt = get_key_of_min_val(nt_err_map);
                }
                else{
                    max_val_nt = get_key_of_max_val(nt_err_map);
                }
                max_val_nt->is_marked = true;
                std::cout<<max_val_nt->neuron_index<<", ";
                nt_err_map.erase(max_val_nt);
            }
        }
    }
    else{
        std::map<Neuron_t*, double>::iterator itr;
        for(itr = nt_err_map.begin(); itr != nt_err_map.end(); itr++){
            itr->first->is_marked = true;
            std::cout<<itr->first->neuron_index<<", ";
        }
    }
    std::cout<<std::endl;

    if(is_marked){
        start_layer->pred_layer->is_marked = true;
        return true;
    }
    return false;
}

void get_marked_neurons_without_maxsat(Network_t* net){
    
    for(Layer_t* layer : net->layer_vec){
        if(layer->is_activation){
            bool is_layer_marked = is_layer_marked_after_optimization_without_maxsat(layer);
            if(is_layer_marked){
                break;
            }
        }
    }
}

bool run_refinement_cegar(Network_t* net){
    bool is_ce = is_sat_val_ce(net);
    if(is_ce){
        return true;
    }

    // get_marked_neurons_reverse(net);
    // return false;

    GRBModel model = create_grb_env_and_model();
    std::vector<GRBVar> var_vector;
    create_milp_mark_milp_refine_constr(net, model, var_vector);

    if(IS_MAXSAT_ANALYSIS){
        get_marked_neurons(model, net, var_vector);
    }
    else{
        get_marked_neurons_without_maxsat(net);
    }

    if(Configuration_deeppoly::is_concurrent){
        for(Layer_t* layer : net->layer_vec){
            for(Neuron_t* nt : layer->neurons){
                if(nt->is_marked){
                    bool is_already_exist = false;
                    for(Neuron_t* nt1 : Global_vars::new_marked_nts){
                        if(nt == nt1){
                            is_already_exist = true;
                        }
                    }
                    if(!is_already_exist){
                        Global_vars::new_marked_nts.push_back(nt);
                    }
                }
            }
        }
    }
    
    return false;
}

bool is_image_verified_by_milp(Network_t* net){
    verify_dim:
    GRBModel model = create_grb_env_and_model();
    std::vector<GRBVar> var_vector;
    create_milp_mark_milp_refine_constr(net, model, var_vector);
    size_t i=0;
    if(Configuration_deeppoly::is_target_ce){
        bool is_verif = verify_by_milp(net, model, var_vector, TARGET_CLASS, true);
        return is_verif;
    }
    for(i=0; i<net->output_dim; i++){
        if(i != net->actual_label){
            bool is_already_verified = false;
            for(size_t val : net->verified_out_dims){
                if(val == i){
                    is_already_verified = true;
                }
            }
            if(!is_already_verified){
                if(!verify_by_milp(net, model, var_vector, i, true)){
                    std::cout<<"Dims: "<<i<<", not verified"<<std::endl;
                    net->counter_class_dim = i;
                    return false;
                }
                else{
                    std::cout<<"Dims: "<<i<<", verified"<<std::endl;
                    net->verified_out_dims.push_back(i);
                    if(Configuration_deeppoly::is_reset_marked_nts){
                        unmark_net(net);
                        goto verify_dim;
                    }
                }
            }
        }
    }
    if(net->verified_out_dims.size() >= 9){ //all output labels verified
        return true;
    }
    std::cout<<"Check................................................................................................"<<std::endl;
    return false;
}

void create_milp_mark_milp_refine_constr(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector){
    creating_vars_milp(net, model, var_vector);
    size_t var_counter = net->input_layer->dims;
    for(auto layer : net->layer_vec){
        if(layer->is_activation){
            create_relu_constr_milp_refine(layer, model, var_vector, var_counter);
        }
        else{
            create_milp_constr_FC_without_marked(layer, model, var_vector, var_counter);
        }
        var_counter += layer->dims;
    }
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

void create_exact_relu_constr_milp_refine(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter){
    assert(layer->is_activation && "Not activation layer\n");
    for(size_t i=0; i< layer->dims; i++){
        std::string contr_name = get_constr_name(layer->layer_index, i);
        Neuron_t* pred_nt = layer->pred_layer->neurons[i];
        if(pred_nt->lb <= 0){
            GRBLinExpr grb_expr = var_vector[var_counter+i] - var_vector[var_counter + i - layer->pred_layer->dims];
            model.addConstr(grb_expr, GRB_EQUAL, 0, contr_name);
        }
        else if(pred_nt->ub <= 0){
            GRBLinExpr grb_expr = var_vector[var_counter + i];
            model.addConstr(grb_expr, GRB_EQUAL, 0, contr_name);
        }
        else{
            create_milp_or_lp_encoding_relu(model, var_vector, var_counter, layer, i, true);
        }

    }
}


void create_relu_constr_milp_refine(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter){
    assert(layer->is_activation && "Not activation layer\n");
    for(size_t i=0; i< layer->dims; i++){
        std::string contr_name = get_constr_name(layer->layer_index, i);
        Neuron_t* pred_nt = layer->pred_layer->neurons[i];
        if(pred_nt->lb <= 0){
            GRBLinExpr grb_expr = var_vector[var_counter+i] - var_vector[var_counter + i - layer->pred_layer->dims];
            model.addConstr(grb_expr, GRB_EQUAL, 0, contr_name);
        }
        else if(pred_nt->ub <= 0){
            GRBLinExpr grb_expr = var_vector[var_counter + i];
            model.addConstr(grb_expr, GRB_EQUAL, 0, contr_name);
        }
        else if(pred_nt->is_marked){
            create_milp_or_lp_encoding_relu(model, var_vector, var_counter, layer, i, true);
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

void create_milp_constr_FC_without_marked(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter){    
    assert(layer->layer_type == "FC" && "Layer type is not FC");
    size_t start_index = var_counter - layer->pred_layer->dims;
    size_t end_index = var_counter;
    std::vector<GRBVar> new_vec;
    new_vec.reserve(layer->pred_layer->dims);
    copy_vector_by_index(var_vector, new_vec, start_index, end_index);
    for(size_t i=0; i<layer->dims; i++){
        std::string constr_name = get_constr_name(layer->layer_index, i);
        Neuron_t* nt = layer->neurons[i];
        GRBLinExpr grb_expr;// = -1*var_vector[end_index+i];
        grb_expr.addTerms(&nt->uexpr->coeff_sup[0], &new_vec[0], new_vec.size());
        grb_expr += nt->uexpr->cst_sup;
        grb_expr += -1*var_vector[var_counter+nt->neuron_index];
        model.addConstr(grb_expr, GRB_EQUAL, 0, constr_name);
    }
}

bool is_prp_verified_by_milp(Network_t* net){
    GRBModel model = create_grb_env_and_model();
    std::vector<GRBVar> var_vector;
    create_milp_mark_milp_refine_constr(net, model, var_vector);
    bool is_sat = is_sat_prop_main_pure_milp(net, model, var_vector);
    return !is_sat;
}







void create_milp_constr_FC_without_marked_ab(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter){    
    assert(layer->layer_type == "FC" && "Layer type is not FC");
    size_t start_index = var_counter - layer->pred_layer->dims;
    size_t end_index = var_counter;
    std::vector<GRBVar> new_vec;
    new_vec.reserve(layer->pred_layer->dims);
    copy_vector_by_index(var_vector, new_vec, start_index, end_index);
    for(size_t i=0; i<layer->dims; i++){
        std::string constr_name = get_constr_name(layer->layer_index, i);
        auto coll = xt::col(layer->w,i);
        std::vector<double> vec(coll.begin(), coll.end());
        double cst = layer->b[i];
        Neuron_t* nt = layer->neurons[i];
        GRBLinExpr grb_expr;// = -1*var_vector[end_index+i];
        grb_expr.addTerms(&vec[0], &new_vec[0], new_vec.size());
        grb_expr += cst;
        grb_expr += -1*var_vector[var_counter+nt->neuron_index];
        model.addConstr(grb_expr, GRB_EQUAL, 0, constr_name);
    }
}



void create_milp_mark_milp_refine_constr_ab(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector){
    creating_vars_milp(net, model, var_vector);
    size_t var_counter = net->input_layer->dims;
    for(auto layer : net->layer_vec){
        if(layer->is_activation){
            create_relu_constr_milp_refine(layer, model, var_vector, var_counter);
        }
        else{
            create_milp_constr_FC_without_marked_ab(layer, model, var_vector, var_counter);
        }
        var_counter += layer->dims;
    }

}

bool is_prp_verified_ab(Network_t* net){
    GRBModel model = create_grb_env_and_model();
    std::vector<GRBVar> var_vector;
    create_milp_mark_milp_refine_constr_ab(net, model, var_vector);
    size_t var_counter = net->input_layer->dims;
    for(size_t i=0; i<net->layer_vec.size()-1; i++){
        var_counter += net->layer_vec[i]->dims;
    }
    Layer_t* last_layer = net->layer_vec.back();
    for(Neuron_t* nt : last_layer->neurons){
        double lb = -nt->lb;
        if(lb <= 0){
            GRBVar var = var_vector[var_counter+nt->neuron_index];
            GRBLinExpr obj = var;
            model.setObjective(obj, GRB_MINIMIZE);
            model.optimize();
            int status = model.get(GRB_IntAttr_Status);
            if(status == GRB_OPTIMAL){
                double obj_val = model.get(GRB_DoubleAttr_ObjVal);
                if(obj_val > 0){
                    nt->lb = -obj_val;
                }
                else{
                    if(obj_val > lb){
                        nt->lb = -obj_val;
                    }
                    update_sat_vals(net,var_vector);
                    nt->sat_val = obj_val;
                    net->dim_under_analysis = nt->neuron_index;
                    return false;
                }
            }
        }
    }
    return true;
}

void print_real_ce(Network_t* net){
    for(size_t i=0; i<net->input_dim; i++){
        std::cout<<net->input_layer->res[i]<<",";
    }
    std::cout<<std::endl;
}

void print_real_ce_status(double conf){
     std::cout<<"Found counter assignment!!"<<std::endl;
    std::cout<<"CE confidence: "<<conf<<std::endl;
}

double compute_softmax_conf(Network_t* net, size_t label){
    Layer_t* last_layer = net->layer_vec.back();
    double label_val = last_layer->res[label] + DIFF_TOLERANCE;
    double denominator = 0;
    for(size_t i=0; i<net->output_dim; i++){
        double val = last_layer->res[i];
        denominator += pow(EULER_C, val);
    }

    double conf = pow(EULER_C, label_val)/denominator;
    return conf;
}

double compute_conf(Network_t* net, size_t label){
    Layer_t* last_layer = net->layer_vec.back();
    double label_val = last_layer->res[label] + DIFF_TOLERANCE;
    double denominator = 0;
    for(size_t i=0; i<net->output_dim; i++){
        denominator += last_layer->res[i];
    }

    double conf = label_val/denominator;
    return conf;
}

bool is_ce_with_softmax_conf(Network_t* net){
    Layer_t* last_layer = net->layer_vec.back();
    double max_val = last_layer->res[net->pred_label] + DIFF_TOLERANCE;
    double denominator = 0;
    for(size_t i=0; i<net->output_dim; i++){
        double val = last_layer->res[i];
        denominator += pow(EULER_C, val);
        if(i != net->pred_label){
            if(max_val < (val + Configuration_deeppoly::softmax_conf_value)){
                return false;
            }
        }
    }

    Global_vars::ce_im_conf = pow(EULER_C, max_val)/denominator;
    return true;
}

bool is_ce_with_conf(Network_t* net){
    Layer_t* last_layer = net->layer_vec.back();
    
    double sum_out = 0;
    for(size_t i=0; i<net->output_dim; i++){
        sum_out += last_layer->res[i];
    }

    double conf = (last_layer->res[net->pred_label])/sum_out;
    if(conf >= Configuration_deeppoly::conf_value){
        Global_vars::ce_im_conf = conf;
        std::cout<<"CE confidence - "<<conf<<std::endl;    
        std::cout<<"Found counter assignment!!"<<" --- "<<pthread_self()<<std::endl;
        return true;
    }
    return false;
}

void assign_net_pred(Network_t* net){
    Layer_t* last_layer = net->layer_vec.back();
    double max_val = last_layer->res[net->actual_label];
    for(size_t i=0; i<net->output_dim; i++){
        double val = last_layer->res[i];
        if(net->actual_label != i){
            if(max_val <= (val+DIFF_TOLERANCE)){
                max_val = val+DIFF_TOLERANCE;
                net->pred_label = i;
            }
        }
    }
}

bool is_sat_val_ce(Network_t* net){
    bool is_counter_example = false;
    create_satvals_to_image(net->input_layer);
    net->forward_propgate_network(0, net->input_layer->res);
    if(Configuration_deeppoly::vnnlib_prp_file_path != ""){
        bool is_sat = is_prop_sat_vnnlib(net);
        return is_sat;
    }

    assign_net_pred(net);

    if(net->actual_label != net->pred_label){
        if(Configuration_deeppoly::is_conf_ce){
            is_counter_example = is_ce_with_conf(net);
        }
        else if(Configuration_deeppoly::is_softmax_conf_ce){
            is_counter_example = is_ce_with_softmax_conf(net);
        }
        else{
            is_counter_example = true;
        }

        if(is_counter_example){
            std::cout<<"CE output values: ";
            Layer_t* last_layer = net->layer_vec.back();
            for(size_t i=0; i<net->output_dim; i++){
                std::cout<<last_layer->res[i]<<" , ";
            }
            std::cout<<std::endl;
        }
    }
    return is_counter_example;
}

bool is_layer_marked_after_optimization(Layer_t* start_layer, std::vector<GRBVar>& var_vector, size_t var_counter){
    bool is_marked = false;
    std::map<Neuron_t*, double> nt_err_map;
    auto res_iter = start_layer->res.begin();
    for(size_t i=0; i<start_layer->neurons.size(); i++, res_iter++){
        Neuron_t* pred_nt = start_layer->pred_layer->neurons[i];
        GRBVar var = var_vector[var_counter+i];
        // std::cout<<"Marking......."<<std::endl;
        double sat_val = var.get(GRB_DoubleAttr_X);
        double res = *res_iter;
        double diff = abs(sat_val - res);
        // std::cout<<diff<<" , "<<sat_val<<" , "<<res<<std::endl;
        if(diff > DIFF_TOLERANCE){
            if(pred_nt->lb > 0 && pred_nt->ub > 0){
                is_marked = true;
                nt_err_map[pred_nt] = diff;
            }
        }
    }
    // std::cout<<std::endl;
    std::cout<<"Layer index: "<<start_layer->pred_layer->layer_index<<", marked neurons: ";
    if(nt_err_map.size() > MAX_NUM_MARKED_NEURONS){
        for(size_t i = 0; i<MAX_NUM_MARKED_NEURONS; i++){
            if(nt_err_map.size() > 0){
                Neuron_t* max_val_nt = NULL;
                if(IS_TOP_MIN_DIFF){
                    max_val_nt = get_key_of_min_val(nt_err_map);
                }
                else{
                    max_val_nt = get_key_of_max_val(nt_err_map);
                }
                max_val_nt->is_marked = true;
                std::cout<<max_val_nt->neuron_index<<", ";
                nt_err_map.erase(max_val_nt);
            }
        }
    }
    else{
        std::map<Neuron_t*, double>::iterator itr;
        for(itr = nt_err_map.begin(); itr != nt_err_map.end(); itr++){
            itr->first->is_marked = true;
            std::cout<<itr->first->neuron_index<<", ";
        }
    }
    std::cout<<std::endl;

    if(is_marked){
        start_layer->pred_layer->is_marked = true;
        return true;
    }
    return false;
}

void update_vars_bounds_by_prev_satval(Layer_t* layer, std::vector<GRBVar>& var_vector, size_t var_counter){
    for(size_t i=0; i<layer->dims; i++){
        GRBVar var = var_vector[var_counter+i];
        Neuron_t* nt = layer->neurons[i];
        var.set(GRB_DoubleAttr_LB, nt->sat_val);
        var.set(GRB_DoubleAttr_UB, nt->sat_val);
    }
}

void update_vars_bounds(Layer_t* layer, std::vector<GRBVar>& var_vector, size_t var_counter){
    for(size_t i=0; i<layer->dims; i++){
        GRBVar var = var_vector[var_counter+i];
        double sat_val = var.get(GRB_DoubleAttr_X);
        var.set(GRB_DoubleAttr_LB, sat_val);
        var.set(GRB_DoubleAttr_UB, sat_val);
    }
}

void remove_maxsat_constr(GRBModel& model, Layer_t* layer){
    for(size_t i=0; i<layer->dims; i++){
        std::string constr_name = get_consr_name_binary(layer->layer_index, i);
        std::string constr_name1 = constr_name+",1";
        std::string constr_name2 = constr_name+",2";
        
        GRBConstr constr = model.getConstrByName(constr_name1);
        model.remove(constr);
        constr = model.getConstrByName(constr_name2);
        model.remove(constr);
    }
    model.update();
}