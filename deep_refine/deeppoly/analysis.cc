#include "analysis.hh"
#include "interval.hh"
#include "helper.hh"
#include "optimizer.hh"
#include "deeppoly_configuration.hh"
#include<thread>
#include<unordered_set>

bool forward_analysis(Network_t* net){
    bool is_verified = false;
    for(auto layer:net->layer_vec){
        if(layer->is_activation){
            if(Configuration_deeppoly::is_parallel){
                forward_layer_ReLU_parallel(net, layer);
            }
            else{
                forward_layer_ReLU(net, layer, 0, layer->dims);
            }
        }
        else{
            if(Configuration_deeppoly::is_parallel){
                forward_layer_FC_parallel(net, layer);
            }
            else{
                forward_layer_FC(net, layer, 0, layer->dims);
            }
        }
    }
    if(Configuration_deeppoly::vnnlib_prp_file_path != ""){
        bool is_sat = is_sat_property_main(net);
        is_verified = !is_sat;
    }
    else{
        is_verified = is_image_verified_deeppoly(net);
    }
    return is_verified;
}


bool milp_based_deeppoly(Network_t* net, Layer_t* marked_layer){
    GRBModel model = create_env_and_model();
    std::vector<GRBVar> var_vector;
    creating_variables_one_layer(net, model, var_vector, net->input_layer);
    size_t var_counter = net->input_layer->dims;
    for(int i=0; i<marked_layer->layer_index; i++){
        Layer_t* layer = net->layer_vec[i];
        creating_variables_one_layer(net, model, var_vector, layer);
        if(layer->is_activation){
            create_milp_constr_relu(layer, model, var_vector, var_counter);
        }
        else{
            create_milp_constr_FC(layer, model, var_vector, var_counter);
        }
        var_counter += layer->dims;
    }
    int numlayer = net->layer_vec.size();
    for(int i=marked_layer->layer_index; i<numlayer; i++){
        Layer_t* layer = net->layer_vec[i];
        bool is_verif = forward_layer_milp(net, layer, model, var_vector, var_counter);
        if(is_verif){
            return true;
        }
        var_counter += layer->dims;
    }

    bool is_verified = is_image_verified_milp(net, model, var_vector);
    return is_verified;
}

bool forward_layer_milp(Network_t* net, Layer_t* layer, GRBModel& model, std::vector<GRBVar>& var_vector, size_t var_counter){
    if(layer->is_activation){
        if(Configuration_deeppoly::is_parallel){
            forward_layer_ReLU_parallel(net, layer);
        }
        else{
            forward_layer_ReLU(net, layer, 0, layer->dims);
        }
        creating_variables_one_layer(net, model, var_vector, layer);
        create_milp_constr_relu(layer, model, var_vector, var_counter);
        return false;
    }
    else{
        creating_variables_one_layer(net, model, var_vector, layer);
        size_t start_index = var_counter - layer->pred_layer->dims;
        size_t end_index = var_counter;
        std::vector<GRBVar> pred_layer_vars;
        pred_layer_vars.reserve(layer->pred_layer->dims);
        copy_vector_by_index(var_vector, pred_layer_vars, start_index, end_index);
        //create_milp_constr_FC(layer, model, var_vector, var_counter);
        bool is_infisiable_path = milp_layer_FC(layer, model, pred_layer_vars, var_vector, var_counter, 0, layer->dims);
        return is_infisiable_path;
        // if(Configuration_deeppoly::is_parallel){
        //     milp_layer_FC_parallel(layer, model, pred_layer_vars, var_vector, var_counter);
        // }
        // else{
        //     milp_layer_FC(layer, model, pred_layer_vars, var_vector, var_counter, 0, layer->dims);
        // }
    }
}

void milp_layer_FC_parallel(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& pred_layer_vars, std::vector<GRBVar>& var_vector, size_t var_counter){
    unsigned int num_thread = get_num_thread();
    std::vector<std::thread> threads;
    size_t num_neurons = layer->dims;
    size_t pool_size = num_neurons/num_thread;
    if(num_neurons < num_thread){
        pool_size = 1;
    }
    size_t start_index = 0;
    size_t end_index = start_index + pool_size;

    for(size_t i=0; i<num_thread; i++){
        threads.push_back(std::thread(milp_layer_FC, layer, std::ref(model), std::ref(pred_layer_vars), std::ref(var_vector), var_counter, start_index, end_index));
        if(end_index >= num_neurons){
            break;
        }
        start_index = end_index;
        end_index = start_index + pool_size;
        if(end_index > num_neurons){
            end_index = num_neurons;
        }
        if(i == num_thread-2){
            end_index = num_neurons;
        }
    }

    for(auto &th : threads){
        th.join();
    }
}

bool milp_layer_FC(Layer_t* layer, GRBModel& model, std::vector<GRBVar>& pred_layer_vars, std::vector<GRBVar>& var_vector, size_t var_counter, size_t start_index, size_t end_index){
    for(size_t i=start_index; i<end_index; i++){
        Neuron_t* nt = layer->neurons[i];
        create_neuron_expr_FC(nt, layer);
        create_milp_constr_FC_node(nt, model, var_vector, pred_layer_vars, var_counter);
        
        GRBLinExpr obj_expr = var_vector[var_counter+nt->neuron_index];
        model.setObjective(obj_expr, GRB_MINIMIZE);
        model.optimize();
        bool is_unsat = set_neurons_bounds(layer, nt, model, true);
        if(is_unsat){
            return true;
        }
        var_vector[var_counter+nt->neuron_index].set(GRB_DoubleAttr_LB, -nt->lb);
        
        model.setObjective(obj_expr, GRB_MAXIMIZE);
        model.optimize();
        is_unsat = set_neurons_bounds(layer, nt, model, false);
        if(is_unsat){
            return true;
        }
        var_vector[var_counter+nt->neuron_index].set(GRB_DoubleAttr_UB, nt->ub);
        //std::cout<<"Layer: "<<layer->layer_index<<" , "<<nt->neuron_index<<" bounds: ["<<-nt->lb<<","<<nt->ub<<"]"<<std::endl;
        //model.reset();
    }
    return false;
}

bool set_neurons_bounds(Layer_t* layer, Neuron_t* nt, GRBModel& model, bool is_lower){
    int status = model.get(GRB_IntAttr_Status);
    if(status == GRB_OPTIMAL){
        if(is_lower){
            double lb = model.get(GRB_DoubleAttr_ObjVal);
            nt->lb = -fmax(lb, -nt->lb);
        }
        else{
            double ub = model.get(GRB_DoubleAttr_ObjVal);
            nt->ub = fmin(ub, nt->ub);
        }
    }
    else if(status == GRB_INFEASIBLE){
        std::cout<<"Layer index: "<<layer->layer_index<<" , neuron index: "<<nt->neuron_index<<": Infisible bounds"<<std::endl;
        return true;
    }
    else if(status == GRB_UNBOUNDED){
        std::cout<<"Layer index: "<<layer->layer_index<<" , neuron index: "<<nt->neuron_index<<": UBOUNDED bounds"<<std::endl;
    }
    else{
        std::cout<<"Layer index: "<<layer->layer_index<<" , neuron index: "<<nt->neuron_index<<": UNKNOWN bounds"<<std::endl;
    }
    return false;
}

void forward_layer_FC_parallel(Network_t* net, Layer_t* curr_layer){
    unsigned int num_thread = get_num_thread();
    std::vector<std::thread> threads;
    size_t num_neurons = curr_layer->dims;
    size_t pool_size = num_neurons/num_thread;
    if(num_neurons < num_thread){
        pool_size = 1;
    }
    size_t start_index = 0;
    size_t end_index = start_index + pool_size;

    for(size_t i=0; i<num_thread; i++){
        threads.push_back(std::thread(forward_layer_FC, net, curr_layer, start_index, end_index));
        if(end_index >= num_neurons){
            break;
        }
        start_index = end_index;
        end_index = start_index + pool_size;
        if(end_index > num_neurons){
            end_index = num_neurons;
        }
        if(i == num_thread-2){
            end_index = num_neurons;
        }
    }

    for(auto &th : threads){
        th.join();
    }

}

void forward_layer_FC(Network_t* net, Layer_t* curr_layer, size_t start_index, size_t end_index){
    assert(curr_layer->layer_type == "FC" && "Not FC layer"); 
    for(size_t i=start_index; i<end_index; i++){
        Neuron_t* nt = curr_layer->neurons[i];
        update_neuron_FC(net, curr_layer, nt);
    }
}

void forward_layer_ReLU_parallel(Network_t* net, Layer_t* curr_layer){
    unsigned int num_thread = get_num_thread();
    std::vector<std::thread> threads;
    size_t num_neurons = curr_layer->dims;
    size_t pool_size = num_neurons/num_thread;
    if(num_neurons < num_thread){
        pool_size = 1;
    }
    size_t start_index = 0;
    size_t end_index = start_index + pool_size;

    for(size_t i=0; i<num_thread; i++){
        threads.push_back(std::thread(forward_layer_ReLU, net, curr_layer, start_index, end_index));
        if(end_index >= num_neurons){
            break;
        }
        start_index = end_index;
        end_index = start_index + pool_size;
        if(end_index > num_neurons){
            end_index = num_neurons;
        }
        if(i == num_thread-2){
            end_index = num_neurons;
        }
    }

    for(auto &th : threads){
        th.join();
    }
}

void forward_layer_ReLU(Network_t* net, Layer_t* curr_layer, size_t start_index, size_t end_index){
    assert(curr_layer->is_activation && "Not a ReLU layer");
    Layer_t* pred_layer = get_pred_layer(net, curr_layer);
    for(size_t i=0; i<curr_layer->dims; i++){
        Neuron_t* nt = curr_layer->neurons[i];
        update_neuron_relu(net, pred_layer, nt);
    }
}

void update_neuron_relu(Network_t* net, Layer_t* pred_layer, Neuron_t* nt){
    Neuron_t* pred_nt = pred_layer->neurons[nt->neuron_index];
    nt->lb = -fmax(0.0, -pred_nt->lb);
    nt->ub = fmax(0.0, pred_nt->ub);
    update_relu_expr(nt, pred_nt, true, true);
    update_relu_expr(nt, pred_nt, true, false);
}

void update_relu_expr(Neuron_t* curr_nt, Neuron_t* pred_nt, bool is_default_heuristic, bool is_lower){
    Expr_t* res_expr = new Expr_t();
    res_expr->size = 1;
    double lb = pred_nt->lb;
	double ub = pred_nt->ub;
	double width = ub + lb;
	double slope_inf = -ub/width;
	double slope_sup = ub/width;
    if(ub<=0){
		res_expr->coeff_inf.push_back(0.0);
		res_expr->coeff_sup.push_back(0.0);
	}
	else if(lb<0){
		res_expr->coeff_inf.push_back(-1.0);
		res_expr->coeff_sup.push_back(1.0);
	}
    else if(is_lower){
        double area1 = 0.5*ub*width;
		double area2 = 0.5*lb*width;
		if(is_default_heuristic){
			if(area1 < area2){
				res_expr->coeff_inf.push_back(0.0);
				res_expr->coeff_sup.push_back(0.0);
			}
			else{
				res_expr->coeff_inf.push_back(-1.0);
				res_expr->coeff_sup.push_back(1.0);
			}
		}
		else{
				res_expr->coeff_inf.push_back(0.0);
				res_expr->coeff_sup.push_back(0.0);
		}
    }
    else{
        double offset_inf = slope_inf*lb;
		double offset_sup = slope_sup*lb;
		res_expr->coeff_inf.push_back(slope_inf);
		res_expr->coeff_sup.push_back(slope_sup);
		res_expr->cst_inf = offset_inf;
		res_expr->cst_sup = offset_sup;
    }

    if(is_lower){
        curr_nt->lexpr = res_expr;
    }
    else{
        curr_nt->uexpr = res_expr;
    }
	
}

void update_neuron_FC(Network_t* net, Layer_t* layer, Neuron_t* nt){
    assert(layer->layer_index >= 0 && "Layer indexing is wrong");
    create_neuron_expr_FC(nt, layer);
    Layer_t* pred_layer = get_pred_layer(net, layer);
    update_neuron_bound_back_substitution(net, pred_layer, nt);
}

void create_neuron_expr_FC(Neuron_t* nt, Layer_t* layer){
    std::vector<size_t> shape =  layer->w_shape;
    nt->uexpr = new Expr_t();
    nt->lexpr = new Expr_t();
    nt->uexpr_b = new Expr_t();
    nt->lexpr_b = new Expr_t();
    nt->uexpr->size = shape[0];
    nt->lexpr->size = shape[0];
    nt->uexpr_b->size = shape[0];
    nt->lexpr_b->size = shape[0];
    nt->uexpr->coeff_inf.resize(nt->uexpr->size);
    nt->uexpr->coeff_sup.resize(nt->uexpr->size);
    nt->lexpr->coeff_inf.resize(nt->lexpr->size);
    nt->lexpr->coeff_sup.resize(nt->lexpr->size);
    nt->uexpr_b->coeff_inf.resize(nt->uexpr_b->size);
    nt->uexpr_b->coeff_sup.resize(nt->uexpr_b->size);
    nt->lexpr_b->coeff_inf.resize(nt->lexpr_b->size);
    nt->lexpr_b->coeff_sup.resize(nt->lexpr_b->size);
    auto coll = xt::col(layer->w,nt->neuron_index);
    //std::cout<<"Column of layer, neuron index: ("<<layer->layer_index<<","<<nt->neuron_index<<") is, "<<coll<<std::endl;
    for(size_t i=0; i < shape[0]; i++){
        double coff = coll[i];//layer->w[i,nt->neuron_index];
        nt->uexpr->coeff_inf[i] = -coff;
        nt->uexpr->coeff_sup[i] = coff;
        nt->lexpr->coeff_inf[i] = -coff;
        nt->lexpr->coeff_sup[i] = coff;
        nt->uexpr_b->coeff_inf[i] = -coff;
        nt->uexpr_b->coeff_sup[i] = coff;
        nt->lexpr_b->coeff_inf[i] = -coff;
        nt->lexpr_b->coeff_sup[i] = coff;
    }
    double cst = layer->b[nt->neuron_index];
    nt->uexpr->cst_inf = -cst;
    nt->uexpr->cst_sup = cst;
    nt->lexpr->cst_inf = -cst;
    nt->lexpr->cst_sup = cst;
    nt->uexpr_b->cst_inf = -cst;
    nt->uexpr_b->cst_sup = cst;
    nt->lexpr_b->cst_inf = -cst;
    nt->lexpr_b->cst_sup = cst;
}



void update_neuron_lexpr_bound_back_substitution(Network_t* net, Layer_t* pred_layer, Neuron_t* nt){
    nt->lb = fmin(nt->lb, compute_lb_from_expr(pred_layer, nt->lexpr_b));
    if(pred_layer->layer_index >= 0){
        if(pred_layer->is_activation){
            Expr_t* tmp_expr_l = update_expr_relu_backsubstitution(net,pred_layer,nt->lexpr_b, nt, true);
            delete nt->lexpr_b;
            nt->lexpr_b = tmp_expr_l;
        }
        else{
            Expr_t* tmp_expr_l = update_expr_affine_backsubstitution(net, pred_layer,nt->lexpr_b,nt,true);
            delete nt->lexpr_b;
            nt->lexpr_b = tmp_expr_l;
        }
        Layer_t* pred_pred_layer = get_pred_layer(net, pred_layer);
        update_neuron_lexpr_bound_back_substitution(net, pred_pred_layer, nt);
    }
}

void update_neuron_uexpr_bound_back_substitution(Network_t* net, Layer_t* pred_layer, Neuron_t* nt){
    nt->ub = fmin(nt->ub, compute_ub_from_expr(pred_layer, nt->uexpr_b));
    if(pred_layer->layer_index >= 0){
        Layer_t* pred_pred_layer = get_pred_layer(net, pred_layer);
        if(pred_layer->is_activation){
            Expr_t* tmp_expr_u = update_expr_relu_backsubstitution(net, pred_layer, nt->uexpr_b, nt, false);
            delete nt->uexpr_b;
            nt->uexpr_b = tmp_expr_u;
        }
        else{
            Expr_t* tmp_expr_u = update_expr_affine_backsubstitution(net, pred_layer, nt->uexpr_b, nt, false);
            delete nt->uexpr_b;
            nt->uexpr_b = tmp_expr_u;
        }
        update_neuron_uexpr_bound_back_substitution(net, pred_pred_layer, nt);
    }
}

void update_neuron_bound_back_substitution(Network_t* net, Layer_t* pred_layer, Neuron_t* nt){
    update_neuron_lexpr_bound_back_substitution(net, pred_layer, nt);
    update_neuron_uexpr_bound_back_substitution(net, pred_layer, nt);
}


Expr_t* update_expr_affine_backsubstitution(Network_t* net, Layer_t* pred_layer,Expr_t* curr_expr, Neuron_t* curr_nt, bool is_lower){
    assert(pred_layer->layer_type == "FC" && "Not FC layer");
    Expr_t* res_expr = new Expr_t();
    Layer_t* pred_pred_layer = get_pred_layer(net,pred_layer);
    res_expr->size = pred_pred_layer->dims;
    res_expr->cst_inf = 0;
    res_expr->cst_sup = 0;
    res_expr->coeff_inf.resize(res_expr->size, 0.0);
    res_expr->coeff_sup.resize(res_expr->size, 0.0);
    Neuron_t* pred_nt = NULL;
    Expr_t* mul_expr = NULL;
    Expr_t* temp_expr = NULL;
    //std::vector<Constr_t*> new_constr_vec;
    //create_constr_vec_with_init_expr(new_constr_vec, curr_expr->constr_vec, res_expr->size);
    for(size_t i=0; i<curr_expr->size; i++){
        pred_nt = pred_layer->neurons[i];
        if(curr_expr->coeff_inf[i] == 0 && curr_expr->coeff_sup[i] == 0){
            //update_independent_constr_FC(net, new_constr_vec, curr_expr->constr_vec, pred_nt);
            continue;
        }
        mul_expr = get_mul_expr(pred_nt, curr_expr->coeff_inf[i], curr_expr->coeff_sup[i], is_lower);
        //update_dependent_constr_FC(net,new_constr_vec, curr_expr->constr_vec, mul_expr, pred_nt);
        if(curr_expr->coeff_inf[i] < 0 || curr_expr->coeff_sup[i] < 0){
            temp_expr = multiply_expr_with_coeff(net, mul_expr, curr_expr->coeff_inf[i], curr_expr->coeff_sup[i]);
            add_expr(net, res_expr, temp_expr);
            delete temp_expr;
        }
        else{
            double temp1, temp2;
			double_interval_mul_cst_coeff(net->ulp, net->min_denormal,&temp1,&temp2,pred_nt->lb,pred_nt->ub,curr_expr->coeff_inf[i],curr_expr->coeff_sup[i]);
			if(is_lower){
				res_expr->cst_inf = res_expr->cst_inf + temp1;
				res_expr->cst_sup = res_expr->cst_sup - temp1;
			}
			else{
				res_expr->cst_inf = res_expr->cst_inf - temp2;
				res_expr->cst_sup = res_expr->cst_sup + temp2;
			}
        }
    }

    res_expr->cst_inf = res_expr->cst_inf + curr_expr->cst_inf;
    res_expr->cst_sup = res_expr->cst_sup + curr_expr->cst_sup;

    return res_expr;
}

Expr_t* update_expr_relu_backsubstitution(Network_t* net, Layer_t* pred_layer, Expr_t* curr_expr, Neuron_t* nt, bool is_lower){
    assert(pred_layer->is_activation && "Not activation layer");
    Expr_t* res_expr = new Expr_t();
    res_expr->size = pred_layer->dims;
    res_expr->cst_inf = curr_expr->cst_inf;
    res_expr->cst_sup = curr_expr->cst_sup;
    res_expr->coeff_inf.resize(res_expr->size);
    res_expr->coeff_sup.resize(res_expr->size);

    //std::vector<Constr_t*> new_constr_vec;
    //create_constr_vec_by_size(new_constr_vec, curr_expr->constr_vec, res_expr->size);

    for(size_t i=0; i<curr_expr->size; i++){
        Neuron_t* pred_nt = pred_layer->neurons[i];
        if((curr_expr->coeff_inf[i] == 0.0) && (curr_expr->coeff_sup[i] == 0.0)){
            res_expr->coeff_inf[i] = 0.0;
            res_expr->coeff_sup[i] = 0.0;
            //update_independent_constr_relu(net,new_constr_vec,curr_expr->constr_vec,pred_nt);
            continue;
        }
        Expr_t* mul_expr = get_mul_expr(pred_nt, curr_expr->coeff_inf[i], 
                                            curr_expr->coeff_sup[i], is_lower);
        //update_dependent_constr_relu(net,new_constr_vec, curr_expr->constr_vec, mul_expr, pred_nt);       
        if(curr_expr->coeff_sup[i] < 0 || curr_expr->coeff_inf[i] < 0){
            /*
            double_interval_mul(&res_expr->coeff_inf[i], &res_expr->coeff_sup[i],
                                            mul_expr->coeff_inf[0], mul_expr->coeff_sup[0],
                                            curr_expr->coeff_inf[i], curr_expr->coeff_sup[i]);
            */
            double_interval_mul_expr_coeff(net->ulp,&res_expr->coeff_inf[i], &res_expr->coeff_sup[i],
                                            mul_expr->coeff_inf[0], mul_expr->coeff_sup[0],
                                            curr_expr->coeff_inf[i], curr_expr->coeff_sup[i]);
            double tmp1, tmp2;
            /*
            double_interval_mul(&tmp1, &tmp2,
                                            mul_expr->cst_inf, mul_expr->cst_sup,
                                            curr_expr->coeff_inf[i], curr_expr->coeff_sup[i]);
            */
            double_interval_mul_cst_coeff(net->ulp, net->min_denormal, &tmp1, &tmp2,
                                            mul_expr->cst_inf, mul_expr->cst_sup,
                                            curr_expr->coeff_inf[i], curr_expr->coeff_sup[i]);
            res_expr->cst_inf = res_expr->cst_inf + tmp1 + net->min_denormal;
            res_expr->cst_sup = res_expr->cst_sup + tmp2 + net->min_denormal;
        }
        else{
            res_expr->coeff_inf[i] = 0.0;
            res_expr->coeff_sup[i] = 0.0;
            double tmp1, tmp2;
            double_interval_mul_expr_coeff(net->ulp, &tmp1,&tmp2, pred_nt->lb, pred_nt->ub, 
                                            curr_expr->coeff_inf[i],curr_expr->coeff_sup[i]);    
            
            if(is_lower){
				res_expr->cst_inf = res_expr->cst_inf + tmp1;
                res_expr->cst_sup = res_expr->cst_sup - tmp1;
			}
			else{
                res_expr->cst_inf = res_expr->cst_inf - tmp2;
                res_expr->cst_sup = res_expr->cst_sup + tmp2;
			}
        }
    }
    //free_constr_vector_memory(curr_expr->constr_vec);
    //res_expr->constr_vec = new_constr_vec;

    return res_expr;
}

void create_input_layer_expr(Network_t* net){
    Layer_t* layer = net->input_layer;
    double ep = Configuration_deeppoly::epsilon;
    for(size_t i=0; i < layer->dims; i++){
        Neuron_t* nt = layer->neurons[i];
        nt->ub = layer->res[i] + ep;
        nt->lb = layer->res[i] - ep;
        if(nt->ub > 1.0){
            nt->ub = 1.0;
        }
        if(nt->lb < 0.0){
            nt->lb = 0.0;
        }
        nt->lb = -nt->lb;
    }
    if(Configuration_deeppoly::is_small_ex){
        for(size_t i=0; i< layer->dims; i++){
            Neuron_t* nt = layer->neurons[i];
            nt->ub = 1;
            nt->lb = 1;//actual is -1
        }
    }
}


double compute_lb_from_expr(Layer_t* pred_layer, Expr_t* expr){
    double res = expr->cst_inf;
    for(size_t i=0; i < expr->size; i++){
        double temp1, temp2;
        double_interval_mul(&temp1, &temp2, expr->coeff_inf[i], expr->coeff_sup[i], 
                                pred_layer->neurons[i]->lb, pred_layer->neurons[i]->ub);
        res += temp1;
    }
    return res;
}

double compute_ub_from_expr(Layer_t* pred_layer, Expr_t* expr){
    double res = expr->cst_sup;
    for(size_t i=0; i < expr->size; i++){
        double temp1, temp2;
        double_interval_mul(&temp1, &temp2, expr->coeff_inf[i], expr->coeff_sup[i], 
                                pred_layer->neurons[i]->lb, pred_layer->neurons[i]->ub);
        res += temp2;
    }
    return res;
}

bool is_no_ce_with_conf(Network_t* net){
    bool is_verified = true;
    Layer_t* out_layer = net->layer_vec.back();
    double denominator = 0.0;
    for(size_t i=0; i<net->output_dim; i++){
        double lb = -out_layer->neurons[i]->lb;
        denominator += lb;
    }
    denominator = CONFIDENCE_OF_CE*denominator;
    for(size_t i=0; i<net->output_dim; i++){
        if(i != net->actual_label){
            double ub = out_layer->neurons[i]->ub;
            std::cout<<"Dim: "<<i<<" , error: "<<(ub - denominator)<<std::endl;
            if(denominator > ub){
                net->verified_out_dims.push_back(i);
            }
            else{
                is_verified = false;
            }
        }
    }

    return is_verified;
}

bool is_image_verified_deeppoly(Network_t* net){
    bool is_verified = true;
    if(IS_CONF_CE){
        is_verified = is_no_ce_with_conf(net);
        return is_verified;
    }
    for(size_t i=0; i<net->output_dim; i++){
        if(i != net->actual_label){
            bool is_already_verified = false;
            for(size_t val : net->verified_out_dims){
                if(val == i){
                    is_already_verified = true;
                }
            }
            if(!is_already_verified){
                if(is_greater(net, net->actual_label, i, true)){
                    net->verified_out_dims.push_back(i);
                }
                else{
                    is_verified = false;
                }
            }
        }
    }
    return is_verified;
}


bool is_image_verified(Network_t* net){
    bool is_verified = true;
    bool is_first= true;
    // net->verified_out_dims.clear();
    std::cout<<"Verified dims: ";
    for(size_t val : net->verified_out_dims){
        std::cout<<val<<" ";
    }
    std::cout<<std::endl;
    std::vector<GRBVar> var_vector;
    GRBModel model = create_env_model_constr(net, var_vector);
    for(size_t i=0; i<net->output_dim; i++){
        if(i != net->actual_label){
            bool is_already_verified = false;
            for(size_t val : net->verified_out_dims){
                if(val == i){
                    is_already_verified = true;
                }
            }
            if(!is_already_verified){
                if(!is_greater(net, net->actual_label, i, true)){
                    if(Configuration_deeppoly::tool == "drefine"){
                        if(!verify_by_milp(net, model, var_vector, i, is_first)){
                            //return false;
                            if(is_first){
                                net->counter_class_dim = i;
                            }
                            is_first = false;
                            is_verified = false;
                        }
                        else{
                            net->verified_out_dims.push_back(i);
                            net->index_map_dims_to_split.erase(i);
                        }
                    }
                    else{
                        net->counter_class_dim = i;
                        return false;
                    }
                }
                else if(Configuration_deeppoly::tool == "drefine"){
                    net->verified_out_dims.push_back(i);
                }
            }
        }
    }
    return is_verified;
}

bool is_greater(Network_t* net, size_t index1, size_t index2, bool is_stricly_greater){
    //return true, if value at index1 is higher than value at index2
    assert(index1 >= 0 && index1 < net->output_dim && "index1 out of bound");
    assert(index2 >= 0 && index2 < net->output_dim && "index2 out of bound");
    Layer_t* out_layer = net->layer_vec.back();
    Neuron_t* nt1 = out_layer->neurons[index1];
    Neuron_t* nt2 = out_layer->neurons[index2];
    Layer_t* pred_layer = new Layer_t();
    pred_layer->neurons = {nt1, nt2};
    pred_layer->dims = 2;
    pred_layer->layer_index = out_layer->layer_index;
    pred_layer->is_activation = out_layer->is_activation;
    pred_layer->layer_type = out_layer->layer_type;
    if(is_stricly_greater){
        if(-nt1->lb > nt2->ub){
            return true;
        }
    }
    else{
        if(-nt1->lb >= nt2->ub){
            return true;
        }
    }
    

    Neuron_t* nt = new Neuron_t();
    nt->lb = INFINITY;
    nt->lexpr_b = new Expr_t();
    nt->lexpr_b->cst_inf = 0.0;
    nt->lexpr_b->cst_sup = 0.0;
    nt->lexpr_b->size=2;
    nt->lexpr_b->coeff_inf = {-1.0,1.0};
    nt->lexpr_b->coeff_sup = {1.0, -1.0};
    update_pred_layer_link(net,pred_layer);
    update_neuron_lexpr_bound_back_substitution(net, pred_layer, nt);
    //std::cout<<index1<<", "<<index2<<", lb: "<<-nt->lb<<std::endl;
    if(is_stricly_greater){
        if(nt->lb < 0){ //lower bound is completely positive
            return true;
        }
    }
    else{
        if(nt->lb <= 0){ 
            return true;
        }
    }

    // std::cout<<"Expr size: "<<nt->lexpr_b->coeff_sup.size()<<std::endl;
    
    std::cout<<"Deeppoly error with ("<<index1<<", "<<index2<<") :"<<nt->lb<<std::endl;
    std::vector<size_t> dims_to_split = get_max_elems_indexes_vec(net, nt->lexpr_b->coeff_sup);
    // std::cout<<"Dims: ";
    // for(size_t val : dims_to_split){
    //     std::cout<<val<<" ";
    // }
    // std::cout<<std::endl;

    // std::cout<<"Coeffs values: ";
    // for(size_t val : dims_to_split){
    //     std::cout<<nt->lexpr_b->coeff_sup[val]<<" ";
    // }
    // std::cout<<std::endl;

    net->index_map_dims_to_split[index2] = dims_to_split;
    return false;
}

bool is_image_verified_milp(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector){
    bool is_first = true;
    for(size_t i=0; i<net->output_dim; i++){
        if(i != net->actual_label){
            bool is_already_verified_dim = false;
            for(size_t val : net->verified_out_dims){
                if(val == i){
                    is_already_verified_dim = true;
                }
            }
            if(!is_already_verified_dim){
                if(!verify_by_milp(net, model, var_vector, i, is_first)){
                    is_first = false;
                    return false;
                }
            }
        }
    }

    return true;
}

bool is_sat_prop_main_pure_milp(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vec){
    VnnLib_t* vnn_lib = net->vnn_lib;
    Vnnlib_post_cond_t* prop = vnn_lib->out_prp;
    bool is_sat = false;
    if(prop->type == "disj"){
        bool is_first = true;
        for(Vnnlib_post_cond_t* prp : prop->comp_prp){
            std::vector<Vnnlib_post_cond_t*> verified_prp = prop->verified_sub_prp;
            if(std::find(verified_prp.begin(), verified_prp.end(), prp) == verified_prp.end()){
                bool is_sat_milp = is_sat_with_milp(net, model, var_vec, prp, is_first);
                if(is_sat_milp){
                    is_first = false;
                    is_sat = true;
                }
                else{
                    prop->verified_sub_prp.push_back(prp);
                }
            }
        }
    }
    else if(prop->type == "conj"){
        bool is_sat_milp = is_sat_with_milp(net, model, var_vec, prop, true);
        return is_sat_milp;
    }

    return is_sat;
}

bool is_sat_property_main(Network_t* net){
    VnnLib_t* vnn_lib = net->vnn_lib;
    Vnnlib_post_cond_t* prop = vnn_lib->out_prp;
    bool is_sat = false;
    std::vector<GRBVar> var_vector;
    GRBModel model = create_env_model_constr(net, var_vector);
    if(prop->type == "disj"){
        bool is_first = true;
        for(Vnnlib_post_cond_t* prp : prop->comp_prp){
            std::vector<Vnnlib_post_cond_t*> verified_prp = prop->verified_sub_prp;
            if(std::find(verified_prp.begin(), verified_prp.end(), prp) == verified_prp.end()){ // if prp not in verified_prp
                bool is_sat_sub = is_sat_property_conj(net, prp);
                if(is_sat_sub){
                    bool is_sat_milp = is_sat_with_milp(net, model, var_vector, prp, is_first);
                    if(is_sat_milp){
                        is_first = false;
                        is_sat = true;
                    }
                    else{
                        prop->verified_sub_prp.push_back(prp);
                    }
                }
                else{
                    prop->verified_sub_prp.push_back(prp);
                }
            }
        }
    }
    else if(prop->type == "conj"){
        bool is_sat_sub = is_sat_property_conj(net, prop);
        if(is_sat_sub){
            bool is_sat_milp = is_sat_with_milp(net, model, var_vector, prop, true);
            if(is_sat_milp){
                return true;
            }
        }
    }
    else{
        std::cout<<"Prop type: "<<prop->type<<std::endl;
        assert(0 && "Unknown property type");
    }

    return is_sat;
}

bool is_sat_with_milp(Network_t* net, GRBModel& model, std::vector<GRBVar>& var_vector, Vnnlib_post_cond_t* conj_cond, bool is_first){
    std::vector<GRBConstr> constr_vec;
    std::unordered_set<size_t> indexes_in_prp;
    for(Basic_post_cond_t* basic_cond : conj_cond->basic_prp){
        if(basic_cond->type == "rel"){
            set_rel_cond_constr(net, model, constr_vec, var_vector, indexes_in_prp, basic_cond);
        }
        else if(basic_cond->type == "basic"){
            set_basic_cond_constr(net, model, constr_vec, var_vector, indexes_in_prp, basic_cond);
        }
        else{
            assert(0 && "Invalid format of conj property");
        }
    }
    model.optimize();
    int status = model.get(GRB_IntAttr_Status);
    if(status == GRB_OPTIMAL){
        if(is_first){
            update_sat_vals(net, var_vector);
        }
        remove_constr_grb_model(model, constr_vec);
        return true;
    }
    else if(status == GRB_INFEASIBLE){
        remove_constr_grb_model(model, constr_vec);
        return false;
    }
    else{
        std::cout<<"Gurobi output: "<<status<<std::endl;
        assert(0 && "Unknown gurobi output");
    }
    return true;
}

void remove_constr_grb_model(GRBModel& model, std::vector<GRBConstr>& constr_vec){
    for(GRBConstr con : constr_vec){
        model.remove(con);
    }
    model.update();
    constr_vec.clear();
}

void set_basic_cond_constr(Network_t* net, GRBModel& model, std::vector<GRBConstr>& constr_vec, std::vector<GRBVar>& var_vector, std::unordered_set<size_t>& indexes_in_prp, Basic_post_cond_t* basic_cond){
    Layer_t* layer = net->layer_vec.back();
    std::string op = basic_cond->op;
    if(is_number(basic_cond->lhs)){
        size_t index = get_var_index(basic_cond->rhs, false);
        indexes_in_prp.insert(index);
        double bound = std::stod(basic_cond->lhs);
        size_t grb_var_index  = get_gurobi_var_index(layer, index);
        GRBLinExpr grb_expr = var_vector[grb_var_index] - bound;
        if(op == "<=" || op == "<"){
            GRBConstr con = model.addConstr(grb_expr, GRB_GREATER_EQUAL, 0);
            constr_vec.push_back(con);
        }
        else{
            GRBConstr con = model.addConstr(grb_expr, GRB_LESS_EQUAL, 0);
            constr_vec.push_back(con);
        }
    }
    else if(is_number(basic_cond->rhs)){
        size_t index = get_var_index(basic_cond->lhs, false);
        indexes_in_prp.insert(index);
        double bound = std::stod(basic_cond->rhs);
        size_t grb_var_index  = get_gurobi_var_index(layer, index);
        GRBLinExpr grb_expr = var_vector[grb_var_index] - bound;
        if(op == "<=" || op == "<"){
            GRBConstr con = model.addConstr(grb_expr, GRB_LESS_EQUAL, 0);
            constr_vec.push_back(con);
        }
        else{
            GRBConstr con = model.addConstr(grb_expr, GRB_GREATER_EQUAL, 0);
            constr_vec.push_back(con);
        }
    }
    else{
        assert(0 && "basic condition in wrong format");
    }
}

void set_rel_cond_constr(Network_t* net, GRBModel& model, std::vector<GRBConstr>& constr_vec, std::vector<GRBVar>& var_vector, std::unordered_set<size_t>& indexes_in_prp, Basic_post_cond_t* basic_cond){
    Layer_t* layer = net->layer_vec.back();
    std::string op = basic_cond->op;
    size_t lhs_index = get_var_index(basic_cond->lhs, false);
    size_t rhs_index = get_var_index(basic_cond->rhs, false);
    indexes_in_prp.insert(lhs_index);
    indexes_in_prp.insert(rhs_index);
    size_t lhs_grb_var_index = get_gurobi_var_index(layer, lhs_index);
    size_t rhs_grb_var_index = get_gurobi_var_index(layer, rhs_index);
    GRBLinExpr grb_expr = var_vector[lhs_grb_var_index] - var_vector[rhs_grb_var_index];
    if(op == "<=" || op == "<"){
        GRBConstr con = model.addConstr(grb_expr, GRB_LESS_EQUAL, 0);
        constr_vec.push_back(con);
    }
    else if(op == ">=" || op == ">"){
        GRBConstr con = model.addConstr(grb_expr, GRB_GREATER_EQUAL, 0);
        constr_vec.push_back(con);
    }
    else{
        assert(0 && "basic condition in wrong format");
    }
}

bool is_sat_property_conj(Network_t* net, Vnnlib_post_cond_t* conj_cond){
    bool is_verified = true;
    for(Basic_post_cond_t* basic_cond : conj_cond->basic_prp){
        if(basic_cond->type == "rel"){
            is_verified = is_verified_neg_rel_property(net, basic_cond);
            if(is_verified){
                return false;
            }
        }
        else if(basic_cond->type == "basic"){
            is_verified = is_verified_neg_basic_property(net, basic_cond);
            if(is_verified){
                return false;
            }
        }
        else{
            std::cout<<basic_cond->lhs<<" "<<basic_cond->op<<" "<<basic_cond->rhs<<std::endl;
            assert(0 && "Wrong property type");
        }
    }
    return true;
}

bool is_verified_neg_rel_property(Network_t* net, Basic_post_cond_t* basic_cond){
    std::string lhs = basic_cond->lhs;
    std::string op = basic_cond->op;
    std::string rhs = basic_cond->rhs;
    op = get_neg_op(op);
    bool is_upper = false;
    bool is_strict_cond = false;
    size_t index_1 = get_var_index(lhs, false);
    size_t index_2 = get_var_index(rhs, false);
    if(op == "<" || op == ">"){
        is_strict_cond = true;
    }
    if(op == "<=" || op == "<"){
        is_upper = true;
    }
    bool is_sat;
    if(is_upper){
        is_sat = is_greater(net, index_2, index_1, is_strict_cond);
    }
    else{
        is_sat = is_greater(net, index_1, index_2, is_strict_cond);
    }

    return is_sat;
}

bool is_verified_neg_basic_property(Network_t* net, Basic_post_cond_t* basic_cond){
    std::string lhs = basic_cond->lhs;
    std::string op = basic_cond->op;
    std::string rhs = basic_cond->rhs;
    op = get_neg_op(op);
    std::string bound_str = "";
    std::string var_str;
    bool is_upper = false;;
    bool is_strict_cond = false;
    if(op == "<" || op == ">"){
        is_strict_cond = true;
    }
    if(op == "<=" || op == "<"){
        is_upper = true;
    }

    if(is_number(lhs)){
        bound_str = lhs;
        var_str = rhs;
    }
    else if(is_number(rhs)){
        bound_str = rhs;
        var_str = lhs;
    }
    else{
        std::cout<<lhs<<" "<<op<<" "<<rhs<<std::endl;
        assert(0 && "non basic property");
        return false;
    }

    double bound = std::stod(bound_str);
    size_t index = get_var_index(var_str, false);
    bool is_verified = is_verified_single_nt_bound(net, index, bound, is_upper, is_strict_cond);
    return is_verified;
}

bool is_verified_single_nt_bound(Network_t* net, size_t nt_index, double bound, bool is_upper, bool is_strict_cond){
    Layer_t* layer = net->layer_vec.back();
    Neuron_t* nt = layer->neurons[nt_index];
    if(is_upper){
        if(is_strict_cond){
            if(nt->ub < bound){
                return true;
            }
        }
        else{
           if(nt->ub <= bound){
                return true;
            } 
        }
    }
    else{
        double lb = -nt->lb;
        if(is_strict_cond){
            if(lb > bound){
                return true;
            }
        }
        else{
            if(lb >= bound){
                return true;
            }
        }
    }
    return false;
}

std::string get_neg_op(std::string op){
    if(op == "<="){
        return ">";
    }
    else if(op == ">="){
        return "<";
    }
    else if(op == "<"){
        return ">=";
    }
    else if(op == ">"){
        return "<=";
    }
    else{
        assert(0 && "wrong operator");
    }
    return "";
}

bool is_prop_sat_vnnlib(Network_t* net){
    bool is_sat;
    VnnLib_t* vnn_lib = net->vnn_lib;
    Vnnlib_post_cond_t* prop = vnn_lib->out_prp;
    if(prop->type == "disj"){
        for(Vnnlib_post_cond_t* prp : prop->comp_prp){
            is_sat = is_prop_sat_vnnlib_conj(net, prp);
            if(is_sat){
                return true;
            }
        }
    }
    else if(prop->type == "conj"){
        is_sat = is_prop_sat_vnnlib_conj(net, prop);
        return is_sat;
    }
    else{
        assert(0 && "invalid type of main property");
    }
    return false;
}

bool is_prop_sat_vnnlib_conj(Network_t* net, Vnnlib_post_cond_t* prop){
    bool is_sat;
    if(prop->type == "conj"){
        for(Basic_post_cond_t* cond : prop->basic_prp){
            if(cond->type == "basic"){
                is_sat = is_basic_prop_sat(net, cond);
                if(!is_sat){
                    return false;
                }
            }
            else if(cond->type == "rel"){
                is_sat = is_rel_prop_sat(net, cond);
                if(!is_sat){
                    return false;
                }
            }
            else{
                assert(0 && "Invalid property type");
            }
        }
    }
    return true;
}

bool is_basic_prop_sat(Network_t* net, Basic_post_cond_t* basic_cond){
    Layer_t* last_layer = net->layer_vec.back();
    if(is_number(basic_cond->lhs)){
        size_t index = get_var_index(basic_cond->rhs, false);
        std::string op = basic_cond->op;
        double bound = std::stod(basic_cond->lhs);
        double nt_val = last_layer->res[index];
        if(op == "<"){
            return bound < nt_val;
        }
        else if(op == "<="){
            return bound <= nt_val;
        }
        else if(op == ">"){
            return bound > nt_val;
        }
        else if(op == ">="){
            return bound >= nt_val;
        }
    }
    else if(is_number(basic_cond->rhs)){
        size_t index = get_var_index(basic_cond->lhs, false);
        std::string op = basic_cond->op;
        double bound = std::stod(basic_cond->rhs);
        double nt_val = last_layer->res[index];
        if(op == "<"){
            return nt_val < bound;
        }
        else if(op == "<="){
            return nt_val <= bound;
        }
        else if(op == ">"){
            return nt_val > bound;
        }
        else if(op == ">="){
            return nt_val >= bound;
        }
    }
    
    assert(0 && "Invalid property");
    return true;
}

bool is_rel_prop_sat(Network_t* net, Basic_post_cond_t* basic_cond){
    Layer_t* last_layer = net->layer_vec.back();
    assert(basic_cond->type == "rel" && "Invalid property");
    size_t lhs_index = get_var_index(basic_cond->lhs, false);
    size_t rhs_index = get_var_index(basic_cond->rhs, false);
    std::string op = basic_cond->op;
    double lhs_val = last_layer->res[lhs_index];
    double rhs_val = last_layer->res[rhs_index];
    if(op == "<"){
        return lhs_val < rhs_val;
    }
    else if(op == "<="){
        return lhs_val <= rhs_val;
    }
    else if(op == ">"){
        return lhs_val > rhs_val;
    }
    else if(op == ">="){
        return lhs_val >= rhs_val;
    }

    assert(0 && "Invalid property operator");
    return false;
}

void print_xt_array(xt::xarray<double> x_arr, size_t size){
    for(size_t i=0; i< size; i++){
        std::cout<<x_arr[i]<<" ";
    }
    std::cout<<std::endl;
}

std::vector<size_t> get_max_elems_indexes_vec(Network_t* net, std::vector<double>& vec){
    std::vector<size_t> index_vec;
    std::vector<double> max_vals_vec;
    Layer_t* input_layer = net->input_layer;
    for(size_t i=0; i<MAX_INPUT_DIMS_TO_SPLIT; i++){
        double max_val = -INFINITY;
        size_t index;
        bool is_updated = false;
        for(size_t j=0; j<vec.size(); j++){
            double val = abs(vec[j]);
            Neuron_t* nt = input_layer->neurons[j];
            double lb = -nt->lb;
            double delta = nt->ub - lb;
            val = delta*val;
            if(val > max_val && !is_val_exist_in_vec_double(val, max_vals_vec)){
                max_val = val;
                index = j;
                is_updated = true;
            }
        }
        if(is_updated){
            index_vec.push_back(index);
            max_vals_vec.push_back(max_val);
        }
    }

    return index_vec;
}


bool is_val_exist_in_vec_double(double val, std::vector<double>& vec){
    for(double val1 : vec){
        if(val == val1){
            return true;
        }
    }

    return false;
}
