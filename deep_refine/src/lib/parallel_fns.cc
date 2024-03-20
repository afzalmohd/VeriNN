#include "concurrent_run.hh"
#include "../../deeppoly/optimizer.hh"
#include "milp_mark.hh"
#include "parallel_fns.hh"
#include "milp_refine.hh"
#include "drefine_driver.hh"
bool verify_by_milp_mine(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector, size_t counter_class_index, bool is_first,std::vector<int>activations){
    // model.update();
    // model.write("debug_original.lp");
    if(terminate_flag==1){
        pthread_exit(NULL);
    }
    Layer_t* layer = net->layer_vec.back();
    size_t actual_class_var_index  = get_gurobi_var_index(layer, net->actual_label);
    size_t counter_class_var_index = get_gurobi_var_index(layer, counter_class_index);
    std::string extra_cons="extra_cons";
    GRBLinExpr grb_obj;
    if(Configuration_deeppoly::is_conf_ce){
        size_t index=get_gurobi_var_index(layer, 0);
        grb_obj =  Configuration_deeppoly::conf_value*(var_vector[index]+var_vector[index+1]+var_vector[index+2]+var_vector[index+3]+var_vector[index+4]+var_vector[index+5]+var_vector[index+6]+var_vector[index+7]+var_vector[index+8]+var_vector[index+9]) - var_vector[counter_class_var_index];
        model.addConstr(var_vector[counter_class_var_index]-var_vector[actual_class_var_index],GRB_GREATER_EQUAL,0,extra_cons);
    }
    else{
        grb_obj = var_vector[actual_class_var_index] - var_vector[counter_class_var_index];
    }
    
    // std::cout<<"verify before opti"<<std::endl;
    model.setObjective(grb_obj, GRB_MINIMIZE);
    model.optimize();
    // std::cout<<"vrif after opti"<<std::endl;
    if(Configuration_deeppoly::is_conf_ce){
        GRBConstr temp = model.getConstrByName(extra_cons);
        model.remove(temp);
    }
    int cnt=0;
    if(model.get(GRB_IntAttr_Status) != GRB_OPTIMAL){
        // std::cout<<"here not opti"<<std::endl;
        return true;
    }
    double obj_val;
    try
    {
        // code that could cause exception
        obj_val = model.get(GRB_DoubleAttr_ObjVal);
    }
    catch (const GRBException &exc)
    {
        // catch anything thrown within try block that derives from std::exception
        cnt++;
        std::cerr << exc.getMessage();
    }
    // double obj_val = model.get(GRB_DoubleAttr_ObjVal);
    // std::cout<<"after obj_val"<<std::endl;
    if(obj_val > 0){
        // std::cout<<"returnong true in obj_val"<<std::endl;
        return true;
    }
    // std::cout<<"cnt----------------------------------------"<<cnt<<std::endl;
    // std::cout<<var_vector[actual_class_var_index].get(GRB_StringAttr_VarName)<<" "<<var_vector[actual_class_var_index].get(GRB_DoubleAttr_X)<<std::endl;
    // std::cout<<var_vector[counter_class_var_index].get(GRB_StringAttr_VarName)<<" "<<var_vector[counter_class_var_index].get(GRB_DoubleAttr_X)<<std::endl;
    if(terminate_flag==1){
        pthread_exit(NULL);
    }
    // std::cout<<"here for blood -- "<<pthread_self()<<std::endl;
    pthread_mutex_lock(&lcked);
    if(terminate_flag==1){
        pthread_mutex_unlock(&lcked);
        pthread_exit(NULL);
    }
    if(is_first){
        update_sat_vals(net, var_vector);
        net->index_vs_err[counter_class_index] = -obj_val;
        bool is_coun_ex = is_sat_val_ce(net);
        // std::cout<<"------------------------------------- "<<std::endl;
        if(is_coun_ex){
            terminate_flag=1; 
            pthread_mutex_unlock(&lcked);
            // std::cout<<"before returning "<<" --- "<<pthread_self()<<std::endl;
            return false;
        }
        else{
            is_refine=true;
            refine_comb=activations;
            pthread_mutex_unlock(&lcked);
            return true;
        }
        // std::cout<<"exitinf is first"<<std::endl;
    }
    // net->index_vs_err[counter_class_index] = -obj_val;
    // std::cout<<"verif end"<<std::endl;
    // refine_comb=activations;
    // std::cout<<"returning false in end"<<std::endl;
    return false;
    
}
bool run_milp_mark_with_milp_refine_mine(Network_t* net){
    bool is_ce = is_sat_val_ce(net);
    if(is_ce){
        return true;
    }
    // std::cout<<"here in refine fn"<<std::endl;
    
    for(size_t i=0; i<net->layer_vec.size();i++){
        Layer_t* layer = net->layer_vec[i];
        bool is_marked=false;
        if(layer->is_activation){
            is_marked = is_layer_marked_mine(net, layer);
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
// bool run_milp_refine_with_milp_mark_input_split_mine(Network_t* net){
//     net->counter_class_dim = net->actual_label;
//     size_t loop_upper_bound = MILP_WITH_MILP_LIMIT;
//     size_t loop_counter = 0;
//     while(loop_counter < loop_upper_bound){
//         std::cout<<"refine loop"<<std::endl;
//         bool is_ce = run_milp_mark_with_milp_refine(net);
//         int cntr=0;
//         if(is_ce){
//             std::cout<<"here in is_ce"<<std::endl;
//             return false;
//         }
//         else{
//             bool is_image_verified = is_image_verified_by_milp(net);
//             // bool is_image_verified = concurrent_exec(net);
//             std::cout<<"is image verified "<<is_image_verified<<std::endl;
//             if(is_image_verified){
//                return true;
//             }
//         }
//         loop_counter++;
//         Global_vars::iter_counts += 1;
//     }
//     return 0; //DUMMY RETURN
// }

int next_marked_index_r= 0;
bool is_layer_marked_mine(Network_t* net, Layer_t* start_layer){
    GRBModel model = create_grb_env_and_model();
    std::vector<GRBVar> var_vector;
    creating_vars_with_constant_vars(net, model, var_vector, start_layer->layer_index);
    size_t var_counter = start_layer->pred_layer->dims;
    int numlayers = net->layer_vec.size();
    for(int layer_index = start_layer->layer_index; layer_index < numlayers; layer_index++){
        Layer_t* layer = net->layer_vec[layer_index];
        if(layer->is_activation){
            next_marked_index_r=relu_constr_mine(layer, model, var_vector, var_counter,refine_comb,next_marked_index_r,Global_vars::new_marked_nts);
        }
        else{
            create_milp_constr_FC_without_marked(layer, model, var_vector, var_counter);
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
    // std::cout<<"status==== "<<status<<std::endl;
    if(status == GRB_OPTIMAL){
        // std::cout<<"Layer index: "<<start_layer->pred_layer->layer_index<<", marked neurons: ";
        for(size_t i=0; i<start_layer->neurons.size(); i++){
            Neuron_t* pred_nt = start_layer->pred_layer->neurons[i];
            GRBVar var = var_vector[var_counter+i];
            double sat_val = var.get(GRB_DoubleAttr_X);
            double res = start_layer->res[i];
            double diff = abs(sat_val - res);
            if(diff > DIFF_TOLERANCE){
                if(pred_nt->lb > 0 && pred_nt->ub > 0){
                    is_marked = true;
                    // std::cout<<pred_nt->neuron_index<<", ";
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
                    // std::cout<<"Here to push neurons "<<std::endl;
                    update_marked_neurons_in_vec(max_val_nt);
                    std::cout<<max_val_nt->neuron_index<<", ";
                    nt_err_map.erase(max_val_nt);
                }
            }
        }
        else{
            std::map<Neuron_t*, double>::iterator itr;
            for(itr = nt_err_map.begin(); itr != nt_err_map.end(); itr++){
                itr->first->is_marked = true;
                // std::cout<<"Here to push neurons "<<std::endl;
                update_marked_neurons_in_vec(itr->first);
                std::cout<<itr->first->neuron_index<<", ";
            }
        }
        std::cout<<std::endl;
    }
    else{
        std::cout<<"Maxsat query: UNSAT"<<std::endl;
        // assert(0 && "Something is wrong\n");
        is_marked = is_layer_marked_after_optimization_without_maxsat(start_layer);
    }
    if(is_marked){
        start_layer->pred_layer->is_marked = true;
        return true;
    }
    return false;
}


bool is_image_verified_softmax_concurrent(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vec, std::vector<int>& activations){  
    Layer_t* out_layer = net->layer_vec.back();
    double l_max_var = -INFINITY;
    double u_max_var = -INFINITY;
    for(size_t i=0; i<net->output_dim; i++){
        if(net->actual_label != i){
            Neuron_t* nt = out_layer->neurons[i];
            double lb = -nt->lb;
            if(l_max_var < lb){
                l_max_var = lb;
            }

            if(u_max_var < nt->ub){
                u_max_var = nt->ub;
            }
        }
    }
    std::vector<GRBVar> bin_var_vec;
    std::string max_var_str = "softmax_max_var_"+std::to_string(out_layer->layer_index);
    GRBVar max_var = model.addVar(l_max_var, u_max_var, 0.0, GRB_CONTINUOUS,max_var_str);
    size_t correct_var_idx = get_gurobi_var_index(out_layer, net->actual_label);
    model.addConstr(max_var - var_vec[correct_var_idx] - Global_vars::soft_max_conf_approx, GRB_GREATER_EQUAL, 0);

    for(size_t i=0; i<net->output_dim; i++){
        if(i != net->actual_label){
            Neuron_t* nt = out_layer->neurons[i];
            if(nt->ub > l_max_var){
                double lb = -nt->lb;
                size_t var_idx = get_gurobi_var_index(out_layer, i);
                std::string var_str = "softmax_bin_"+std::to_string(out_layer->layer_index)+"_"+std::to_string(i);
                GRBVar bin_var = model.addVar(0,1,0,GRB_BINARY, var_str);
                bin_var_vec.push_back(bin_var);
                double umax_i = get_umax_i(out_layer, i);
                GRBLinExpr grb_expr1 = max_var - var_vec[var_idx] - (1-bin_var)*(umax_i - lb);
                model.addConstr(grb_expr1, GRB_LESS_EQUAL, 0);
                GRBLinExpr grb_expr2 = max_var - var_vec[var_idx] - (1-bin_var)*Global_vars::soft_max_conf_approx;
                model.addConstr(grb_expr2, GRB_GREATER_EQUAL, 0);
            }
        }
    }
    GRBLinExpr sum_expr = 0;
    for(GRBVar var : bin_var_vec){
        sum_expr += var;
    }
    model.addConstr(sum_expr, GRB_EQUAL, 1);

    if(terminate_flag==1){
        pthread_exit(NULL);
    }

    // size_t idx = get_gurobi_var_index(out_layer, 0);

    // for(size_t i=0; i<net->output_dim; i++){
    //     if(i != 6){
    //         model.addConstr(var_vec[idx+6]-var_vec[idx+i]-Configuration_deeppoly::softmax_conf_value, GRB_GREATER_EQUAL, 0);
    //     }
    // }

    // std::string model_file_path = "/home/u1411251/jawwad/code/VeriNN/deep_refine";
    // model_file_path += "/model.lp";
    // model.write(model_file_path);

    model.optimize();

    int status = model.get(GRB_IntAttr_Status);
    std::cout<<"Optimization status: "<<status<<std::endl;
    if(status == GRB_OPTIMAL){
        pthread_mutex_lock(&lcked);
        if(terminate_flag==1){
            pthread_mutex_unlock(&lcked);
            pthread_exit(NULL);
        }
        update_sat_vals(net, var_vec);
        bool is_coun_ex = is_sat_val_ce(net);
        if(is_coun_ex){
            pthread_mutex_unlock(&lcked);
            verif_result = false;
            terminate_flag = true;
            // std::cout<<"real ce"<<std::endl;
            return false;
        }
        //spurius counter example
        // std::cout<<"spurious ce"<<std::endl;
        is_refine=true;
        refine_comb = activations;
        pthread_mutex_unlock(&lcked);
        return true;
    }

    return true;
}
