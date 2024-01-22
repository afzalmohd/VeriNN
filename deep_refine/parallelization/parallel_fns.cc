#include "concurrent_run.hh"
#include "../deeppoly/optimizer.hh"
#include "../src/lib/milp_mark.hh"
#include "parallel_fns.hh"
#include "../src/lib/milp_refine.hh"
#include "../src/lib/drefine_driver.hh"
bool verify_by_milp_mine(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector, size_t counter_class_index, bool is_first,std::vector<int>activations){
    // model.update();
    // model.write("debug_original.lp");
    if(terminate_flag==1){
        pthread_exit(NULL);
    }
    Layer_t* layer = net->layer_vec.back();
    size_t actual_class_var_index  = get_gurobi_var_index(layer, net->actual_label);
    size_t counter_class_var_index = get_gurobi_var_index(layer, counter_class_index);
    GRBLinExpr grb_obj;
    if(Configuration_deeppoly::is_conf_ce){
        size_t index=get_gurobi_var_index(layer, 0);
        grb_obj =  Configuration_deeppoly::conf_of_ce*(var_vector[index]+var_vector[index+1]+var_vector[index+2]+var_vector[index+3]+var_vector[index+4]+var_vector[index+5]+var_vector[index+6]+var_vector[index+7]+var_vector[index+8]+var_vector[index+9]) - var_vector[counter_class_var_index];
    }
    else{
        grb_obj = var_vector[actual_class_var_index] - var_vector[counter_class_var_index];
    }
    
    // std::cout<<"verify before opti"<<std::endl;
    model.setObjective(grb_obj, GRB_MINIMIZE);
    model.optimize();
    // std::cout<<"vrif after opti"<<std::endl;
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
        // std::cout<<"MILP error with ("<<net->actual_label<<","<<counter_class_index<<"): "<<-obj_val<<std::endl;
        // std::cout<<"inside is first"<<std::endl;
        
        // Neuron_t* nt_actual = layer->neurons[net->actual_label];
        // Neuron_t* nt_counter = layer->neurons[counter_class_index];
        // nt_actual->is_back_prop_active = true;
        // nt_actual->back_prop_lb = var_vector[actual_class_var_index].get(GRB_DoubleAttr_X);
        // nt_actual->back_prop_ub = nt_actual->back_prop_lb;
        // nt_counter->is_back_prop_active = true;
        // nt_counter->back_prop_lb = var_vector[counter_class_var_index].get(GRB_DoubleAttr_X);
        // nt_counter->back_prop_ub = nt_counter->back_prop_lb;
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
bool run_milp_refine_with_milp_mark_input_split_mine(Network_t* net){
    net->counter_class_dim = net->actual_label;
    size_t loop_upper_bound = MILP_WITH_MILP_LIMIT;
    size_t loop_counter = 0;
    while(loop_counter < loop_upper_bound){
        std::cout<<"refine loop"<<std::endl;
        bool is_ce = run_milp_mark_with_milp_refine(net);
        int cntr=0;
        if(is_ce){
            std::cout<<"here in is_ce"<<std::endl;
            return false;
        }
        else{
            bool is_image_verified = is_image_verified_by_milp(net);
            // bool is_image_verified = concurrent_exec(net);
            std::cout<<"is image verified "<<is_image_verified<<std::endl;
            if(is_image_verified){
               return true;
            }
        }
        loop_counter++;
        ITER_COUNTS += 1;
    }
    return 0; //DUMMY RETURN
}

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
            //create_relu_constr(layer, model, var_vector, var_counter);
            // create_relu_constr_milp_refine(layer, model, var_vector, var_counter);
            // std::cout<<"New_list mn size = "<<new_list_mn.size()<<std::endl;
            // std::cout<<"refine mn size = "<<refine_comb.size()<<std::endl;
            next_marked_index_r=relu_constr_mine(layer, model, var_vector, var_counter,refine_comb,next_marked_index_r,new_list_mn);
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
                    new_list_mn.push_back(max_val_nt);
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
                new_list_mn.push_back(itr->first);
                std::cout<<itr->first->neuron_index<<", ";
            }
        }
        std::cout<<std::endl;
    }
    else{
        std::cout<<"here in the wrong part"<<std::endl;
        // assert(0 && "Something is wrong\n");
    }
    if(is_marked){
        start_layer->pred_layer->is_marked = true;
        return true;
    }
    return false;
}