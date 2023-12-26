#include "findk.hh"
#include "../src/lib/drefine_driver.hh"
#include "../deeppoly/deeppoly_driver.hh"
#include "gurobi_c++.h"
#include "../src/lib/milp_mark.hh"
#include "../src/lib/milp_refine.hh"
#include "../deeppoly/optimizer.hh"

void set_full_range_input_bounds(Layer_t* layer){
    for(Neuron_t* nt : layer->neurons){
        nt->lb = -0.0;
        nt->ub = 1.0;
    }
}

void copy_neurons_of_layers(Layer_t* layer1, Layer_t* layer2){
    assert(layer1->dims == layer2->dims && "Layers indexes mismatch\n");
    for(size_t i=0; i<layer1->dims; i++){
        Neuron_t* nt1 = layer1->neurons[i];
        Neuron_t* nt2 = layer2->neurons[i];
        nt2->lb = nt1->lb;
        nt2->ub = nt1->ub;
        nt2->is_active = nt1->is_active;
        nt2->is_marked = nt1->is_marked;
        nt2->layer_index = nt1->layer_index;
        nt2->neuron_index = nt1->neuron_index;
        nt2->sat_val = nt1->sat_val;
        nt2->lexpr = nt1->lexpr;
        nt2->lexpr_b = nt1->lexpr_b;
        nt2->uexpr = nt1->uexpr;
        nt2->uexpr_b = nt1->uexpr_b;
    }
}

void copy_neurons_of_networks(Network_t* net1, Network_t* net2){
    copy_neurons_of_layers(net1->input_layer, net2->input_layer);
    for(size_t i=0; i<net1->layer_vec.size(); i++){
        copy_neurons_of_layers(net1->layer_vec[i], net2->layer_vec[i]);
    }
}

void create_input_constr(GRBModel& model, std::vector<GRBVar>& var_vector1, std::vector<GRBVar>& var_vector2, size_t input_dim, double ep){
    for(size_t i=0; i<input_dim; i++){
        GRBVar var1 = var_vector1[i];
        GRBVar var2 = var_vector2[i];
        GRBLinExpr exp = var1 - var2;
        model.addConstr(exp, GRB_LESS_EQUAL, ep);
        model.addConstr(exp, GRB_GREATER_EQUAL, -ep);
    }
}

void create_all_relu_constr_exact(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter){
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

void create_model_constr(GRBModel& model, Network_t* net , std::vector<GRBVar>& var_vector, size_t var_counter){
    for(Layer_t* layer : net->layer_vec){
        if(layer->is_activation){
            create_all_relu_constr_exact(layer, model, var_vector, var_counter);
        }
        else{
            create_milp_constr_FC_without_marked(layer, model, var_vector, var_counter);
        }
        var_counter += layer->dims;
    }
}

void create_output_constr(GRBModel& model, Network_t* net1, std::vector<GRBVar>& var_vector1, std::vector<GRBVar>& var_vector2, size_t var_counter, double k){
    double large_const = 10e5;
    for(size_t i=0; i<net1->output_dim; i++){
        GRBVar x1 = var_vector1[var_counter + i];
        GRBVar x2 = var_vector2[var_counter + i];
        std::string var_str = "bin_k_"+std::to_string(i);
        GRBVar bin_var = model.addVar(0, 1, 0.0, GRB_BINARY, var_str);
        std::string constr_name = get_constr_name(net1->layer_vec.size()-1, i);
        std::string constr_name1 = constr_name+"_k_1";
        std::string constr_name2 = constr_name+"_k_2";
        GRBLinExpr exp1 = x1 - x2 - k*bin_var + large_const*(1- bin_var);
        GRBLinExpr exp2 = x1 - x2 +(1-bin_var)*k - bin_var*large_const;
        model.addConstr(exp1, GRB_GREATER_EQUAL, 0, constr_name1);
        model.addConstr(exp2, GRB_LESS_EQUAL, 0, constr_name2);
    }
}

void remove_output_constr(GRBModel& model, Network_t* net1, std::vector<GRBVar>& var_vector1, std::vector<GRBVar>& var_vector2, size_t var_counter){
    for(size_t i=0; i<net1->output_dim; i++){
        std::string var_str = "bin_k_"+std::to_string(i);
        std::string constr_name = get_constr_name(net1->layer_vec.size()-1, i);
        std::string constr_name1 = constr_name+"_k_1";
        std::string constr_name2 = constr_name+"_k_2";
        GRBVar var = model.getVarByName(var_str);
        GRBConstr constr1 = model.getConstrByName(constr_name1);
        GRBConstr constr2 = model.getConstrByName(constr_name2);
        model.remove(constr1);
        model.remove(constr2);
        model.remove(var);
        model.update();
    }
}

void create_output_constraints_optimization(GRBModel& model, Network_t* net1, std::vector<GRBVar>& var_vector1, std::vector<GRBVar>& var_vector2, size_t var_counter){
    double k=0.5;
    double k_lb = 0.0;
    double k_ub = 0.0;
    double prev_k = 100.0;
    double k_diff_tol = 1e-2;
    size_t bounds_for_k = 10;
    bool is_found_k = false;
    size_t i=0;
    for(i=0; i<bounds_for_k; i++){
        create_output_constr(model, net1, var_vector1, var_vector2, var_counter, k);
        GRBLinExpr obj_exp = 0.0;
        model.setObjective(obj_exp, GRB_MAXIMIZE);
        std::cout<<"Optimizing for k: "<<k<<std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        model.optimize();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration<double>(end_time - start_time);
        int status = model.get(GRB_IntAttr_Status);
        std::cout<<"Model status: "<<status<<", value of (prev_k,k): ("<<prev_k<<","<<k<<"), time: "<<total_time.count()<<std::endl;
        prev_k = k;
        if(status == GRB_OPTIMAL){
            if(k_ub == 0.0){
                k_lb = k;
                k += 0.5;
            }
            else{
                k_lb = k;
                k = (k + k_ub)/2;
            }
        }
        else{
            k_ub = k;
            k = (k_lb + k)/2;
        }
        double diff = std::abs(prev_k - k);
        if(diff <= k_diff_tol){
            is_found_k = true;
            break;
        }

        remove_output_constr(model, net1, var_vector1, var_vector2, var_counter);
    }

    if(!is_found_k){
        std::cout<<"Loop bounds exceeded"<<std::endl;
    }
    else{
        std::cout<<"Found value of K in iteration: "<<i+1<<std::endl;
        std::cout<<"Value of K: "<<prev_k<<std::endl;
    }
}

void create_constrs(Network_t* net1, Network_t* net2){
    size_t var_counter = net1->input_dim;
    GRBModel model = create_grb_env_and_model();
    std::vector<GRBVar> var_vector1;
    creating_vars_milp(net1, model, var_vector1);
    std::vector<GRBVar> var_vector2;
    creating_vars_milp(net2, model, var_vector2);

    create_input_constr(model, var_vector1, var_vector2, net1->input_dim, Configuration_deeppoly::epsilon);

    create_model_constr(model, net1, var_vector1, var_counter);

    create_model_constr(model, net2, var_vector2, var_counter);

    for(Layer_t* layer : net1->layer_vec){
        var_counter += layer->dims;
    }

    var_counter -= net1->output_dim;

    create_output_constraints_optimization(model, net1, var_vector1, var_vector2, var_counter);

    for(size_t i=0; i<net1->output_dim; i++){
        GRBVar x1 = var_vector1[var_counter+i];
        GRBVar x2 = var_vector2[var_counter+i];
        std::cout<<"Vals: "<<x1.get(GRB_DoubleAttr_X)<<" , "<<x2.get(GRB_DoubleAttr_X)<<std::endl;
    }
    


}

bool is_differ_bounds_layers(Layer_t* layer1, Layer_t* layer2){
    for(size_t i=0; i<layer1->dims; i++){
        Neuron_t* nt1 = layer1->neurons[i];
        Neuron_t* nt2 = layer2->neurons[i];
        std::cout<<-nt1->lb<<" , "<<nt1->ub<<std::endl;
        if(nt1->lb != nt2->lb || nt1->ub != nt2->ub){
            return true;
        }
    }

    return false;
}

void is_bounds_differs(Network_t* net1, Network_t* net2){
    bool is_differ = is_differ_bounds_layers(net1->input_layer, net2->input_layer);
    std::cout<<"Layer index: "<<net1->input_layer->layer_index<<" , "<<is_differ<<std::endl;
    for(size_t i=0; i<net1->layer_vec.size(); i++){
        is_differ = is_differ_bounds_layers(net1->layer_vec[i], net2->layer_vec[i]);
        std::cout<<"Layer index: "<<i<<" , "<<is_differ<<std::endl;
    }
}


void find_k(){
    Network_t* net1 = deeppoly_initialize_network();
    set_full_range_input_bounds(net1->input_layer);
    run_deeppoly(net1);
    Network_t* net2 = new Network_t();
    copy_network(net2, net1);
    copy_neurons_of_networks(net1, net2);

    // is_bounds_differs(net1, net2);
    

    create_constrs(net1, net2);

}