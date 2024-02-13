#include "bounds_milp.hh"
#include "../../deeppoly/optimizer.hh"
#include "../../deeppoly/deeppoly_configuration.hh"
#include "milp_mark.hh"
#include "milp_refine.hh"
#include<thread>

std::vector<GRBModel> VEC_OF_MODEL;
std::vector<std::vector<GRBVar>> VEC_OF_VAR_VEC;

bool is_verified_by_bound_tighten_milp(Network_t* net){
    bool is_verified = false;
    forward_analysis_bounds_milp_seq(net);
    is_verified = is_image_verified_by_milp(net);
    if(is_verified){
         std::cout<<"..................Verified by bound tightening................."<<std::endl;
    }
   
    return is_verified;
}

void create_layer_constraints(Network_t* net, Layer_t* layer, size_t var_counter){
    if(layer->layer_index == -1){
        for(size_t i=0; i<NUM_THREADS; i++){
            GRBModel model = create_grb_env_and_model();
            std::vector<GRBVar> var_vector;
            create_vars_layer(layer, model, var_vector);
            VEC_OF_MODEL.push_back(model);
            VEC_OF_VAR_VEC.push_back(var_vector);
        }
    }
    else{
        for(size_t i=0; i<NUM_THREADS; i++){
            GRBModel& model = VEC_OF_MODEL[i];
            std::vector<GRBVar>& var_vector = VEC_OF_VAR_VEC[i];
            create_vars_layer(layer, model, var_vector);
            std::cout<<"Vars created for layer........ "<<layer->layer_index<<std::endl;
            if(layer->is_activation){
                create_exact_relu_constr_milp_refine(layer, model, var_vector, var_counter);
            }
            else{
                create_milp_constr_FC_without_marked(layer, model, var_vector, var_counter);
            }
        }
    }
}

void update_bounds_nt(GRBModel& model, Layer_t* layer, Neuron_t* nt, std::vector<GRBVar>& var_vector){
    size_t milp_var_idx = get_gurobi_var_index(layer, nt->neuron_index);
    std::string nt_name = "nt_"+std::to_string(layer->layer_index)+"_"+std::to_string(nt->neuron_index);
    //find lower bound
    GRBVar var = var_vector[milp_var_idx];
    GRBLinExpr expr1 = var;
    double time_limit = (layer->layer_index+1)*TIME_LIMIT_BOUND_TIGHTNING;
    model.set(GRB_DoubleParam_TimeLimit, time_limit);
    model.setObjective(expr1, GRB_MINIMIZE);
    model.optimize();
    int status = model.get(GRB_IntAttr_Status);
    IFVERBOSE(std::cout<<nt_name<<": lb model status: "<<status<<std::endl);
    if(status == GRB_OPTIMAL){
        try{
            double val = var.get(GRB_DoubleAttr_X);
            double lb = -nt->lb;
            if(val > lb){
                nt->lb = -val;
                var.set(GRB_DoubleAttr_LB, -nt->lb);
            }
        }
        catch(GRBException e){
            std::cout<<"Exception in GRB"<<std::endl;
        }
    }

    // model.reset();
    //find lower bound
    // GRBVar var = var_vector[milp_var_idx];
    GRBLinExpr expr2 = var;
    model.setObjective(expr2, GRB_MAXIMIZE);
    model.optimize();
    status = model.get(GRB_IntAttr_Status);
    IFVERBOSE(std::cout<<nt_name<<": ub model status: "<<status<<std::endl);
    if(status == GRB_OPTIMAL){
        try{
            double val = var.get(GRB_DoubleAttr_X);
            if(val < nt->ub){
                nt->ub = val;
                var.set(GRB_DoubleAttr_UB, nt->ub);
            }
        }
        catch(GRBException e){
            std::cout<<"Exception in GRB"<<std::endl;
        }
    }
}

void update_bounds(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector){
    for(Neuron_t* nt : layer->neurons){
        update_bounds_nt(model, layer, nt, var_vector);
    }
}

void bounds_tighten_for_one_layer_one_thread(Network_t* net, Layer_t* layer, size_t start_idx, size_t end_idx){
    // GRBModel model = create_grb_env_and_model();
    GRBEnv env = GRBEnv(true);
    // env.set("LogFile", "milp.log");
    env.start();
    GRBModel model = GRBModel(env);
    model.set(GRB_IntParam_LogToConsole, 0);
    model.set(GRB_IntParam_OutputFlag, 1);
    model.set(GRB_IntParam_Threads,NUM_GUROBI_THREAD);
    std::vector<GRBVar> var_vector;
    create_vars_layer(net->input_layer, model, var_vector);
    size_t var_counter = net->input_dim;
    for(int i=0; i <= layer->layer_index; i++){
        Layer_t* curr_layer = net->layer_vec[i];
        create_vars_layer(curr_layer, model, var_vector);
        if(curr_layer->is_activation){
            create_exact_relu_constr_milp_refine(curr_layer, model, var_vector, var_counter);
        }
        else{
            create_milp_constr_FC_without_marked(curr_layer, model, var_vector, var_counter);
        }

        var_counter += curr_layer->dims;
    }
    for(size_t i=start_idx; i<end_idx; i++){
        Neuron_t* nt = layer->neurons[i];
        update_bounds_nt(model, layer, nt, var_vector);
    }


}

// void bounds_tighten_for_one_layer_one_thread(Network_t* net, Layer_t* layer, size_t var_counter, size_t start_idx, size_t end_idx, size_t th_seq){
//     GRBModel& model = VEC_OF_MODEL[th_seq];
//     std::vector<GRBVar>& var_vector = VEC_OF_VAR_VEC[th_seq];
//     for(size_t i=start_idx; i<end_idx; i++){
//         Neuron_t* nt = layer->neurons[i];
//         update_bounds_nt(model, layer, nt, var_vector, var_counter);
//     }


// }

// void bounds_tighten_for_one_layer(Network_t* net, Layer_t* layer, size_t var_counter){
//     unsigned int num_thread = NUM_THREADS;
//     std::vector<std::thread> threads;
//     size_t num_neurons = layer->dims;
    
//     std::vector<size_t> loads_per_cpu;
//     if(num_neurons <= num_thread){
//         for(size_t i=0; i<num_neurons; i++){
//             loads_per_cpu.push_back(1);
//         }
//     }
//     else{
//         for(size_t i=0; i<num_thread; i++){
//             loads_per_cpu.push_back(0);
//         }
//         int j=0;
//         for(size_t i=0; i<num_neurons; i++){
//             j =  i % num_thread;
//             loads_per_cpu[j] += 1;
//         }
//     }

//     size_t start_index = 0;
//     for(size_t i=0; i<loads_per_cpu.size(); i++){
//         size_t load = loads_per_cpu[i];
//         threads.push_back(std::thread(bounds_tighten_for_one_layer_one_thread, net, layer, var_counter, start_index, start_index+load, i));
//         start_index += load;
//     }

//     for(auto &th : threads){
//         th.join();
//     }

// }

// void forward_analysis_bounds_milp_parallel(Network_t* net){
//     size_t var_counter = 0;
//     create_layer_constraints(net, net->input_layer, var_counter);
//     var_counter += net->input_dim;
//     for(Layer_t* layer : net->layer_vec){
//         std::cout<<"Bounds tightning for layer........ "<<layer->layer_index<<std::endl;
//         create_layer_constraints(net, layer, var_counter);
//         bounds_tighten_for_one_layer(net, layer, var_counter);
//         var_counter += layer->dims;
//     }
// }

void bounds_tighten_for_one_layer(Network_t* net, Layer_t* layer){
    unsigned int num_thread = NUM_THREADS;
    std::vector<std::thread> threads;
    size_t num_neurons = layer->dims;
    
    std::vector<size_t> loads_per_cpu;
    if(num_neurons <= num_thread){
        for(size_t i=0; i<num_neurons; i++){
            loads_per_cpu.push_back(1);
        }
    }
    else{
        for(size_t i=0; i<num_thread; i++){
            loads_per_cpu.push_back(0);
        }
        int j=0;
        for(size_t i=0; i<num_neurons; i++){
            j =  i % num_thread;
            loads_per_cpu[j] += 1;
        }
    }

    for(size_t i=0; i<loads_per_cpu.size(); i++){
        std::cout<<"Loads on cpu: "<<i<<" , load: "<<loads_per_cpu[i]<<std::endl;
    }
    
    size_t start_index = 0;
    for(size_t i=0; i<loads_per_cpu.size(); i++){
        size_t load = loads_per_cpu[i];
        threads.push_back(std::thread(bounds_tighten_for_one_layer_one_thread, net, layer, start_index, start_index+load));
        start_index += load;
    }

    for(auto &th : threads){
        th.join();
    }

}

void forward_analysis_bounds_milp_parallel(Network_t* net){
    for(Layer_t* layer : net->layer_vec){
        bounds_tighten_for_one_layer(net, layer);
    }
}



void forward_analysis_bounds_milp_seq(Network_t* net){
    GRBModel model = create_grb_env_and_model();
    std::vector<GRBVar> var_vector;
    create_vars_layer(net->input_layer, model, var_vector);
    size_t var_counter = net->input_dim;
    for(size_t i=0; i<net->layer_vec.size() && i <= LAYER_INDEX_UPTO_BOUND_TIGHTEN; i++){
        Layer_t* layer = net->layer_vec[i];
        create_vars_layer(layer, model, var_vector);
        if(layer->is_activation){
            create_exact_relu_constr_milp_refine(layer, model, var_vector, var_counter);
        }
        else{
            create_milp_constr_FC_without_marked(layer, model, var_vector, var_counter);
        }

        update_bounds(layer, model, var_vector);

        var_counter += layer->dims;
    }

    
}


void bounds_tighting_by_milp(Network_t* net){
    if(Configuration_deeppoly::is_parallel){
        forward_analysis_bounds_milp_parallel(net);
    }
    else{
        forward_analysis_bounds_milp_seq(net);
    }
}