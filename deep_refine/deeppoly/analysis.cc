#include "analysis.hh"
#include "interval.hh"
#include "helper.hh"
#include "optimizer.hh"
#include "deeppoly_configuration.hh"
#include<thread>

void forward_analysis(Network_t* net){
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
            create_marked_layer_splitting_constraints(layer);
            if(Configuration_deeppoly::is_parallel){
                forward_layer_FC_parallel(net, layer);
            }
            else{
                forward_layer_FC(net, layer, 0, layer->dims);
            }
        }
    }
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
    if(layer->is_marked){
        copy_layer_constraints(layer, nt);
    }
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
    else{
        if(nt->lexpr_b->constr_vec.size() > 0){
            //printf("Check..l\n");
            compute_bounds_using_gurobi(net, net->layer_vec[pred_layer->layer_index+1], nt, nt->lexpr_b, true);
        }
        if(Configuration_deeppoly::is_unmarked_deeppoly){
            nt->unmarked_lb = nt->lb;
        }
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
    else{
        if(nt->uexpr_b->constr_vec.size() > 0){
            //printf("Check..u\n");
            compute_bounds_using_gurobi(net, net->layer_vec[pred_layer->layer_index+1], nt, nt->uexpr_b, false);
        }
        if(Configuration_deeppoly::is_unmarked_deeppoly){
            nt->unmarked_ub = nt->ub;
        }

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
    std::vector<Constr_t*> new_constr_vec;
    create_constr_vec_with_init_expr(new_constr_vec, curr_expr->constr_vec, res_expr->size);
    for(size_t i=0; i<curr_expr->size; i++){
        pred_nt = pred_layer->neurons[i];
        if(curr_expr->coeff_inf[i] == 0 && curr_expr->coeff_sup[i] == 0){
            update_independent_constr_FC(net, new_constr_vec, curr_expr->constr_vec, pred_nt);
            continue;
        }
        mul_expr = get_mul_expr(pred_nt, curr_expr->coeff_inf[i], curr_expr->coeff_sup[i], is_lower);
        update_dependent_constr_FC(net,new_constr_vec, curr_expr->constr_vec, mul_expr, pred_nt);
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
    update_constr_vec_cst(new_constr_vec, curr_expr->constr_vec);
    free_constr_vector_memory(curr_expr->constr_vec);
    res_expr->constr_vec = new_constr_vec;
    if(pred_layer->is_marked){
        for(auto con : pred_layer->constr_vec){
            Constr_t* constr = new Constr_t();
            constr->expr = new Expr_t();
            constr->deep_copy(con);
            res_expr->constr_vec.push_back(constr);
        }
    }

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

    std::vector<Constr_t*> new_constr_vec;
    create_constr_vec_by_size(new_constr_vec, curr_expr->constr_vec, res_expr->size);

    for(size_t i=0; i<curr_expr->size; i++){
        Neuron_t* pred_nt = pred_layer->neurons[i];
        if((curr_expr->coeff_inf[i] == 0.0) && (curr_expr->coeff_sup[i] == 0.0)){
            res_expr->coeff_inf[i] = 0.0;
            res_expr->coeff_sup[i] = 0.0;
            update_independent_constr_relu(net,new_constr_vec,curr_expr->constr_vec,pred_nt);
            continue;
        }
        Expr_t* mul_expr = get_mul_expr(pred_nt, curr_expr->coeff_inf[i], 
                                            curr_expr->coeff_sup[i], is_lower);
        update_dependent_constr_relu(net,new_constr_vec, curr_expr->constr_vec, mul_expr, pred_nt);       
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
    free_constr_vector_memory(curr_expr->constr_vec);
    res_expr->constr_vec = new_constr_vec;

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


bool is_image_verified(Network_t* net){
    std::vector<GRBVar> var_vector;
    GRBModel model = create_env_model_constr(net, var_vector);
    for(size_t i=0; i<net->output_dim; i++){
        if(i != net->actual_label){
            if(!is_greater(net, net->actual_label, i)){
                if(!verify_by_milp(net, model, var_vector, i)){
                    return false;
                }
            }
        }
    }
    return true;
}

bool is_greater(Network_t* net, size_t index1, size_t index2){
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
    if(-nt1->lb > nt2->ub){
        return true;
    }
    else{
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
        if(nt->lb < 0){ //lower bound is completely positive
            return true;
        }
        std::cout<<"Deeppoly error with ("<<index1<<", "<<index2<<") :"<<nt->lb<<std::endl;
    }

    return false;
}
