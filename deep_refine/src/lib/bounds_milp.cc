#include "bounds_milp.hh"
#include "../../deeppoly/optimizer.hh"
#include "../../deeppoly/deeppoly_configuration.hh"
#include "milp_mark.hh"
#include "milp_refine.hh"


bool is_verified_by_bound_tighten_milp(Network_t* net){
    bool is_verified = false;
    forward_analysis_bounds_milp_seq(net);
    is_verified = is_image_verified_by_milp(net);
    if(is_verified){
         std::cout<<"..................Verified by bound tightening................."<<std::endl;
    }
   
    return is_verified;
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

        update_bounds(layer, model, var_vector, var_counter);

        var_counter += layer->dims;
    }

    
}

void update_bounds(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter){
    for(Neuron_t* nt : layer->neurons){
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
                }
            }
            catch(GRBException e){
                std::cout<<"Exception in GRB"<<std::endl;
            }
        }

        // model.reset();

    }
}