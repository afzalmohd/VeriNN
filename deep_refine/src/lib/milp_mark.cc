#include "milp_mark.hh"
#include "milp_refine.hh"
#include "pullback.hh"
#include "../../deeppoly/optimizer.hh"
#include "../../deeppoly/deeppoly_configuration.hh"
#include "../../deeppoly/analysis.hh"
#include <cstdlib>
#include<map>


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
        assert(0 && "Something is wrong\n");
    }
    if(is_marked){
        start_layer->pred_layer->is_marked = true;
        return true;
    }
    return false;
}

void create_optimization_constraints_layer(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter){
    GRBLinExpr obj=0;
    for(size_t i=0; i<layer->dims; i++){
        Neuron_t* nt = layer->neurons[i];
        std::string var_str = "b,"+std::to_string(layer->layer_index)+","+std::to_string(nt->neuron_index);
        GRBVar bin_var = model.addVar(0, 1, 0.0, GRB_BINARY, var_str);
        model.addGenConstrIndicator(bin_var, 1, var_vector[var_counter+i], GRB_EQUAL, layer->res[i]);
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
    if(layer->layer_index == -1 || layer->layer_index == numlayer-1){//input or output layer
        for(Neuron_t* nt : layer->neurons){
            std::string var_str = "x,"+std::to_string(layer->layer_index)+","+std::to_string(nt->neuron_index);
            GRBVar x = model.addVar(nt->sat_val, nt->sat_val, 0.0, GRB_CONTINUOUS, var_str);
            var_vector.push_back(x);
        }
    }
    else{
        for(size_t i=0; i<layer->dims; i++){
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


bool is_sat_val_ce(Network_t* net){
    create_satvals_to_image(net->input_layer);
    //std::cout<<net->input_layer->res[683]<<" "<<net->input_layer->res[684]<<std::endl;
    net->forward_propgate_network(0, net->input_layer->res);
    if(Configuration_deeppoly::vnnlib_prp_file_path != ""){
        bool is_sat = is_prop_sat_vnnlib(net);
        // if(is_sat){
        //     std::cout<<"input values"<<std::endl;
        //     print_xt_array(net->input_layer->res, net->input_dim);
        //     std::cout<<"Check...."<<std::endl;
        //     std::cout<<net->input_layer->res[683]<<" "<<net->input_layer->res[684]<<std::endl;
        //     std::cout<<"output values"<<std::endl;
        //     print_xt_array(net->layer_vec.back()->res, net->output_dim);
        // }
        return is_sat;
    }
    auto pred_label = xt::argmax(net->layer_vec.back()->res);
    net->pred_label = pred_label[0];
    if(net->actual_label != net->pred_label){
        std::cout<<"Found counter assignment!!"<<std::endl;
        return true;
    }
    return false;
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

// void create_negate_property(GRBModel& model, std::vector<GRBVar>& var_vector, Network_t* net, Layer_t* curr_layer){
//     size_t var_counter = curr_layer->pred_layer->dims;
//     int numlayer = net->layer_vec.size();
//     for(int i=curr_layer->layer_index; i<numlayer-1; i++){
//         var_counter += net->layer_vec[i]->dims;
//     }
//     GRBLinExpr grb_expr = var_vector[var_counter + net->counter_class_dim] - var_vector[var_counter + net->actual_label];
//     model.addConstr(grb_expr, GRB_GREATER_EQUAL, 0);
// }