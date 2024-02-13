#include "milp_mark.hh"
#include "milp_refine.hh"
#include "../../deeppoly/optimizer.hh"
#include "../../deeppoly/deeppoly_configuration.hh"
#include "../../deeppoly/analysis.hh"
#include <cstdlib>
#include<map>

GRBModel create_grb_env_and_model(){
    GRBEnv env = GRBEnv(true);
    env.set("LogFile", "milp.log");
    env.start();
    GRBModel model = GRBModel(env);
    model.set(GRB_IntParam_LogToConsole, 0);
    model.set(GRB_IntParam_OutputFlag, 1);
    model.set(GRB_IntParam_Threads,Configuration_deeppoly::grb_num_thread);
    return model;
}

void creating_constraints(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector){
    size_t var_counter = net->input_dim;
    int numlayers = net->layer_vec.size();
    for(int layer_index = 0; layer_index < numlayers; layer_index++){
        Layer_t* layer = net->layer_vec[layer_index];
        if(layer->is_activation){
            create_relu_constr_milp_refine(layer, model, var_vector, var_counter);
        }
        else{
            if(Configuration_deeppoly::bounds_path != ""){
                create_milp_constr_FC_without_marked_ab(layer, model, var_vector, var_counter);
            }
            else{
                create_milp_constr_FC_without_marked(layer, model, var_vector, var_counter);
            }
        }
        var_counter += layer->dims;
    }
}



void updated_vars_bounds_by_exact_val(Layer_t* layer, std::vector<GRBVar>& var_vector, size_t var_counter){
    auto iter = layer->res.begin();
    for(size_t i=0; i<layer->dims; i++, iter++){
        GRBVar var = var_vector[var_counter+i];
        var.set(GRB_DoubleAttr_LB, *iter);
        var.set(GRB_DoubleAttr_UB, *iter);
    }
}

void updated_vars_bounds_by_original_bounds(Layer_t* layer, std::vector<GRBVar>& var_vector, size_t var_counter){
    for(size_t i=0; i<layer->dims; i++){
        Neuron_t* nt = layer->neurons[i];
        GRBVar var = var_vector[var_counter+i];
        var.set(GRB_DoubleAttr_LB, -nt->lb);
        var.set(GRB_DoubleAttr_UB, nt->ub);
    }
}




void remove_constrs(GRBModel& model, Layer_t* layer){
    if(layer->is_activation){
        for(size_t i=0; i<layer->dims; i++){
            Neuron_t* pred_nt = layer->pred_layer->neurons[i];
            Neuron_t* nt = layer->neurons[i];
            std::string constr_name = get_constr_name(layer->layer_index, i);
            std::string constr_name1 = constr_name+",1";
            std::string constr_name2 = constr_name+",2";
            std::string constr_name3 = constr_name+",3";
            std::string constr_name4 = constr_name+",4";
            if(nt->lb <= 0 || nt->ub <= 0){
                GRBConstr constr = model.getConstrByName(constr_name);
                model.remove(constr);
            }
            else if(pred_nt->is_marked){
                GRBConstr constr = model.getConstrByName(constr_name1);
                model.remove(constr);
                constr = model.getConstrByName(constr_name2);
                model.remove(constr);
                constr = model.getConstrByName(constr_name3);
                model.remove(constr);
                constr = model.getConstrByName(constr_name4);
                model.remove(constr);
            }
            else{
                GRBConstr constr = model.getConstrByName(constr_name1);
                model.remove(constr);
                constr = model.getConstrByName(constr_name2);
                model.remove(constr);
            }
        }
    }
    else{
        for(size_t i=0; i<layer->dims; i++){
            std::string constr_name = get_constr_name(layer->layer_index, i);
            GRBConstr constr = model.getConstrByName(constr_name);
            model.remove(constr);
        }
    }

    model.update();
}


bool is_found_cex_in_abs_exec(GRBModel& model){
    model.optimize();
    int status = model.get(GRB_IntAttr_Status);
    if(status == GRB_OPTIMAL){
        return true;
    }
    else if(status == GRB_INFEASIBLE){
        std::cout<<"Infeasible..........."<<std::endl;
        return false;
    }
    else{
        std::cout<<"Something wrong: grb status: "<<status<<std::endl;
        assert(0);
        return false;
    }
}

void get_marked_nts_after_spurious_cex_reverse(Network_t* net, GRBModel& model, size_t layer_idx, std::vector<GRBVar>& var_vector, size_t var_counter){
    bool is_marked = false;
    for(size_t i=layer_idx; i<net->layer_vec.size(); i++){
        Layer_t* layer = net->layer_vec[i];
        if(layer->is_activation){
            create_optimization_constraints_layer(layer, model, var_vector, var_counter);
            model.optimize();
            int status = model.get(GRB_IntAttr_Status);
            std::cout<<"Optimization status: "<<status<<std::endl;
            bool is_layer_marked = is_layer_marked_after_optimization(layer, var_vector, var_counter);
            if(is_layer_marked){
                is_marked  =true;
                break;
            }
        }
        var_counter += layer->dims;
    }
    if(!is_marked){
        std::cout<<"Something wroong in get_marked_after_spurious_cex_api...."<<std::endl;
        assert(0);
    }
}

void get_marked_neurons_reverse(Network_t* net){
    GRBModel model = create_grb_env_and_model();
    std::vector<GRBVar> var_vector;
    creating_vars_milp(net, model, var_vector);
    size_t var_counter = var_vector.size();
    GRBLinExpr neg_prp_expr = var_vector[var_counter - net->output_dim + net->actual_label] - var_vector[var_counter - net->output_dim + net->counter_class_dim];
    std::string neg_prp_name = "neg_prp";
    model.addConstr(neg_prp_expr, GRB_LESS_EQUAL, 0, neg_prp_name);
    bool is_found_marked_nts = false;
    for(int i=net->layer_vec.size() -1; i>=0; i--){
        Layer_t* layer = net->layer_vec[i];
        var_counter -= layer->dims;
        if(layer->is_activation){
            create_relu_constr_milp_refine(layer, model, var_vector, var_counter);
        }
        else{
            updated_vars_bounds_by_exact_val(layer, var_vector, var_counter);
            bool is_spurious_cex = is_found_cex_in_abs_exec(model);
            if(is_spurious_cex){
                std::cout<<"Found counter at layer........................ : "<<i<<std::endl;
                get_marked_nts_after_spurious_cex_reverse(net, model, layer->layer_index, var_vector, var_counter);
                is_found_marked_nts = true;
                break;
            }
            else{
                updated_vars_bounds_by_original_bounds(layer, var_vector, var_counter);
            }
            create_milp_constr_FC_without_marked(layer, model, var_vector, var_counter);
        }
    }
    if(!is_found_marked_nts){
        std::cout<<"Something wrong!!!!!!!!!!"<<std::endl;
        assert(0);
    }
}


bool run_milp_mark_with_milp_refine(Network_t* net){
    bool is_ce = is_sat_val_ce(net);
    if(is_ce){
        return true;
    }
    
    for(size_t i=0; i<net->layer_vec.size();i++){
        Layer_t* layer = net->layer_vec[i];
        bool is_marked=false;
        if(layer->is_activation){
            is_marked = is_layer_marked(net, layer);
            if(is_marked){
                //std::cout<<"Layer: "<<layer->layer_index<<" marked"<<std::endl;
                break;
            }
        }
        else{
            Layer_t* pred_layer = layer->pred_layer;
            net->forward_propgate_one_layer(layer->layer_index, pred_layer->res);
        }
    }
    return false;
}

bool mark_neurons_with_light_analysis(Network_t* net){
    bool is_ce = is_sat_val_ce(net);
    if(is_ce){
        return true;
    }

    for(size_t i=0; i<MAX_NUM_MARKED_NEURONS; i++){
        double max_val = -INFINITY;
        size_t marked_layer_idx;
        size_t marked_nt_idx;
        for(size_t j=0; j<net->numlayers; j++){
            Layer_t* layer = net->layer_vec[j];
            if(layer->is_activation){
                for(size_t k=0; k<layer->dims; k++){
                    Neuron_t* nt = layer->neurons[k];
                    Neuron_t* pred_nt = layer->pred_layer->neurons[k];
                    double diff = abs(nt->sat_val - layer->res[k]);
                    if(diff > max_val && diff != 0 && (!pred_nt->is_marked)){
                        max_val = diff;
                        marked_layer_idx = j;
                        marked_nt_idx = k;
                    }
                }
            }
        }
        if(max_val != -INFINITY){
            Layer_t* marked_layer = net->layer_vec[marked_layer_idx];
            marked_layer->pred_layer->is_marked = true;
            marked_layer->pred_layer->neurons[marked_nt_idx]->is_marked = true;
            std::cout<<"Layer index, neuron_index, diff: ("<<marked_layer_idx-1<<","<<marked_nt_idx<<","<<max_val<<")"<<std::endl;
        }
    }

    return false;
}

Neuron_t* get_key_of_max_val(std::map<Neuron_t*, double> & m){
    assert(m.size() > 0 && "Map is empty");
    std::map<Neuron_t*, double>::iterator itr;
    bool is_first = true;
    double max_val;
    Neuron_t* max_val_key = NULL;
    for(itr = m.begin(); itr != m.end(); itr++){
        if(is_first){
            max_val_key = itr->first;
            max_val = itr->second;
            is_first = false;
        }
        else{
            if(max_val < itr->second){
                max_val_key = itr->first;
                max_val = itr->second;
            }
        }
    }

    return max_val_key;
}

Neuron_t* get_key_of_min_val(std::map<Neuron_t*, double> & m){
    assert(m.size() > 0 && "Map is empty");
    std::map<Neuron_t*, double>::iterator itr;
    bool is_first = true;
    double min_val;
    Neuron_t* min_val_key = NULL;
    for(itr = m.begin(); itr != m.end(); itr++){
        if(is_first){
            min_val_key = itr->first;
            min_val = itr->second;
            is_first = false;
        }
        else{
            if(min_val > itr->second){
                min_val_key = itr->first;
                min_val = itr->second;
            }
        }
    }

    return min_val_key;
}

bool is_layer_marked(Network_t* net, Layer_t* start_layer){
    GRBModel model = create_grb_env_and_model();
    std::vector<GRBVar> var_vector;
    creating_vars_with_constant_vars(net, model, var_vector, start_layer->layer_index);
    size_t var_counter = start_layer->pred_layer->dims;
    int numlayers = net->layer_vec.size();
    for(int layer_index = start_layer->layer_index; layer_index < numlayers; layer_index++){
        Layer_t* layer = net->layer_vec[layer_index];
        if(layer->is_activation){
            //create_relu_constr(layer, model, var_vector, var_counter);
            create_relu_constr_milp_refine(layer, model, var_vector, var_counter);
        }
        else{
            if(Configuration_deeppoly::bounds_path != ""){
                create_milp_constr_FC_without_marked_ab(layer, model, var_vector, var_counter);
            }
            else{
                create_milp_constr_FC_without_marked(layer, model, var_vector, var_counter);
            }
        }
        var_counter += layer->dims;
    }
    
    //create_negate_property(model, var_vector, net, start_layer);

    var_counter = start_layer->pred_layer->dims;
    create_optimization_constraints_layer(start_layer, model, var_vector, var_counter);
    model.optimize();
    bool is_marked = false;
    int status = model.get(GRB_IntAttr_Status);
    std::map<Neuron_t*, double> nt_err_map;
    if(status == GRB_OPTIMAL){
        // std::cout<<"Layer index: "<<start_layer->pred_layer->layer_index<<", marked neurons: ";
        auto res_iter = start_layer->res.begin();
        for(size_t i=0; i<start_layer->neurons.size(); i++, res_iter++){
            Neuron_t* pred_nt = start_layer->pred_layer->neurons[i];
            GRBVar var = var_vector[var_counter+i];
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
                    Neuron_t* max_val_nt = get_key_of_max_val(nt_err_map);
                    max_val_nt->is_marked = true;
                    Global_vars::new_marked_nts.push_back(max_val_nt);
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
    }
    else{
        std::cout<<"Something wrong in maxsat... Gurobi status: "<<status<<std::endl;
        assert(0 && "Something is wrong\n");
    }
    if(is_marked){
        start_layer->pred_layer->is_marked = true;
        return true;
    }
    else{
        for(size_t i=0; i<start_layer->dims; i++){
            Neuron_t* nt = start_layer->neurons[i];
            GRBVar var = var_vector[var_counter+i];
            double sat_val = var.get(GRB_DoubleAttr_X);
            nt->sat_val = sat_val;
        }
    }
    return false;
}

std::string get_consr_name_binary(size_t layer_idx, size_t nt_idx){
    std::string constr_name = "cb_"+std::to_string(layer_idx)+","+std::to_string(nt_idx);
    return constr_name;
}

void create_optimization_constraints_layer(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter){
    auto iter = layer->res.begin();
    GRBLinExpr obj=0;
    for(size_t i=0; i<layer->dims; i++, iter++){
        std::string constr_name = get_consr_name_binary(layer->layer_index, i);
        std::string constr_name1 = constr_name+",1";
        std::string constr_name2 = constr_name+",2";
        Neuron_t* nt = layer->neurons[i];
        double lb = -nt->lb;
        double range = nt->ub - lb;
        double exact_val = *iter;
        GRBVar var = var_vector[var_counter+i];
        
        std::string var_str = "b,"+std::to_string(layer->layer_index)+","+std::to_string(nt->neuron_index);
        GRBVar bin_var = model.addVar(0, 1, 0.0, GRB_INTEGER, var_str);
        
        GRBLinExpr expr1 = var - exact_val;
        expr1 -= (1-bin_var)*range;
        model.addConstr(expr1, GRB_LESS_EQUAL, 0, constr_name1);
        // std::cout<<"node"<<i<<": var_"<<std::to_string(i)<<"-"<<std::to_string(exact_val)<<" <= "<<"(1-"<<var_str<<")*("<<std::to_string(nt->ub)<<"-"<<std::to_string(lb)<<")"<<std::endl;
        GRBLinExpr expr2 = var - exact_val;
        expr2 += (1-bin_var)*range;
        model.addConstr(expr2, GRB_GREATER_EQUAL, 0, constr_name2);
        // std::cout<<"node"<<i<<": var_"<<std::to_string(i)<<"-"<<std::to_string(exact_val)<<" >= "<<"-(1-"<<var_str<<")*("<<std::to_string(nt->ub)<<"-"<<std::to_string(lb)<<")"<<std::endl;
        obj += bin_var;
    }
    model.setObjective(obj, GRB_MAXIMIZE);
}

void creating_vars_with_constant_vars(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector, size_t start_layer_index){ 
    if(start_layer_index == 0){
        create_constant_vars_satval_layer(net, net->input_layer, model, var_vector);
    }
    else{
        create_constant_vars_satval_layer(net, net->layer_vec[start_layer_index-1], model, var_vector);
    }
    
    size_t layer_index;
    for(layer_index = start_layer_index; layer_index < net->layer_vec.size()-1; layer_index++){
        Layer_t* layer = net->layer_vec[layer_index];
        create_vars_layer(layer, model, var_vector);
    }
    Layer_t* out_layer = net->layer_vec[layer_index];
    create_constant_vars_satval_layer(net,out_layer, model, var_vector);
}



void create_constant_vars_satval_layer(Network_t* net, Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector){
    int numlayer = net->layer_vec.size();
    if(Configuration_deeppoly::bounds_path != "" && layer->layer_index == numlayer-1){
        for(Neuron_t* nt : layer->neurons){
            std::string var_str = "x,"+std::to_string(layer->layer_index)+","+std::to_string(nt->neuron_index);
            if(nt->neuron_index == net->dim_under_analysis){
                GRBVar x = model.addVar(nt->sat_val, nt->sat_val, 0.0, GRB_CONTINUOUS, var_str);
                var_vector.push_back(x);
            }
            else{
                GRBVar x = model.addVar(-nt->lb, nt->ub, 0.0, GRB_CONTINUOUS, var_str);
                var_vector.push_back(x);
            }
        }
    }
    else if(layer->layer_index == -1 || layer->layer_index == numlayer-1){//input or output layer
        for(Neuron_t* nt : layer->neurons){
            std::string var_str = "x,"+std::to_string(layer->layer_index)+","+std::to_string(nt->neuron_index);
            GRBVar x = model.addVar(nt->sat_val, nt->sat_val, 0.0, GRB_CONTINUOUS, var_str);
            var_vector.push_back(x);
        }
    }
    else{
        auto iter = layer->res.begin();
        for(size_t i=0; i<layer->dims; i++, iter++){
            Neuron_t* nt = layer->neurons[i];
            std::string var_str = "x,"+std::to_string(layer->layer_index)+","+std::to_string(nt->neuron_index);
            GRBVar x = model.addVar(layer->res[i], layer->res[i], 0.0, GRB_CONTINUOUS, var_str);
            var_vector.push_back(x);
        }
    }
    
}

void create_relu_constr(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter){
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



void create_satvals_to_image(Layer_t* layer){
    std::vector<double> vec;
    vec.reserve(layer->dims);
    for(size_t i=0; i<layer->dims; i++){
        Neuron_t* nt = layer->neurons[i];
        vec.push_back(nt->sat_val);
    }
    std::vector<size_t> shape = {layer->dims};
    layer->res = xt::adapt(vec,shape);
}

void get_images_from_satval(xt::xarray<double>& res, Layer_t* layer){
    std::vector<double> vec;
    vec.reserve(layer->dims);
    for(size_t i=0; i<layer->dims; i++){
        Neuron_t* nt = layer->neurons[i];
        vec.push_back(nt->sat_val);
    }
    std::vector<size_t> shape = {layer->dims};
    res = xt::adapt(vec,shape);
}
