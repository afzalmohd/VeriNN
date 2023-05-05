#include "helper.hh"
#include "interval.hh"
#include "deeppoly_configuration.hh"
#include<thread>
#include<xtensor/xio.hpp>

unsigned int get_num_thread(){
    unsigned int num_system_cores = std::thread::hardware_concurrency();
    if(num_system_cores < Configuration_deeppoly::num_thread){
        return num_system_cores;
    }
    return Configuration_deeppoly::num_thread;
}

// void copy_layer_constraints(Layer_t* layer, Neuron_t* nt){
//     if(layer->is_marked){
//         for(auto constr : layer->constr_vec){
//             Constr_t* con = new Constr_t();
//             con->expr = new Expr_t();
//             con->deep_copy(constr);
//             nt->uexpr_b->constr_vec.push_back(con);
            
//             Constr_t* con1 = new Constr_t();
//             con1->expr = new Expr_t();
//             con1->deep_copy(constr);
//             nt->lexpr_b->constr_vec.push_back(con1);
//         }
//     }
//     else{
//         IFVERBOSE(std::cout<<"Either layer or neuron is not marked"<<std::endl);
//     }
// }

void update_pred_layer_link(Network_t* net, Layer_t* pred_layer){
    if(pred_layer->is_activation){
        Layer_t* pred_pred_layer = new Layer_t();
        Layer_t* curr_pred_pred_layer = NULL;
        if(pred_layer->layer_index > 0){
            curr_pred_pred_layer = net->layer_vec[pred_layer->layer_index-1];
        }
        else{
            curr_pred_pred_layer = net->input_layer;
        }
        pred_pred_layer->layer_index = curr_pred_pred_layer->layer_index;
        pred_pred_layer->dims = pred_layer->dims;
        pred_pred_layer->activation = curr_pred_pred_layer->activation;
        pred_pred_layer->is_activation = curr_pred_pred_layer->is_activation;
        pred_pred_layer->layer_type = curr_pred_pred_layer->layer_type;
        pred_pred_layer->layer_index = curr_pred_pred_layer->layer_index;
        pred_pred_layer->pred_layer = curr_pred_pred_layer->pred_layer;
        pred_pred_layer->neurons.resize(pred_layer->dims);
        for(size_t i=0; i<pred_layer->dims; i++){
            size_t index = pred_layer->neurons[i]->neuron_index;
            pred_pred_layer->neurons[i] = curr_pred_pred_layer->neurons[index];
        }
        
        pred_layer->pred_layer = pred_pred_layer;
    }
    else{
        if(pred_layer->layer_index > 0){
            pred_layer->pred_layer = net->layer_vec[pred_layer->layer_index -1];
        }
        else{
            pred_layer->pred_layer = net->input_layer;
        }
    }
}

void add_expr(Network_t* net, Expr_t* expr1, Expr_t* expr2){
    assert(expr1->size == expr2->size && "Expression size mismatch while adding");
    double max1 = fmax(fabs(expr1->cst_inf),fabs(expr1->cst_sup));
	double max2 = fmax(fabs(expr2->cst_inf),fabs(expr2->cst_sup));
    expr1->cst_inf += expr2->cst_inf  + (max1 + max2)*net->ulp + net->min_denormal;
	expr1->cst_sup += expr2->cst_sup  + (max1 + max2)*net->ulp + net->min_denormal;
    for(size_t i=0; i<expr1->size; i++){
        max1 = fmax(fabs(expr1->coeff_inf[i]),fabs(expr1->coeff_sup[i]));
		max2 = fmax(fabs(expr2->coeff_inf[i]),fabs(expr2->coeff_sup[i]));
		expr1->coeff_inf[i] = expr1->coeff_inf[i] + expr2->coeff_inf[i] + (max1 + max1)*net->ulp;
		expr1->coeff_sup[i] = expr1->coeff_sup[i] + expr2->coeff_sup[i] + (max1 + max2)*net->ulp;
    }
}

Expr_t* multiply_expr_with_coeff(Network_t* net, Expr_t* expr, double coeff_inf, double coeff_sup){
    Expr_t* mul_expr = new Expr_t();
    mul_expr->size = expr->size;
    mul_expr->coeff_inf.resize(mul_expr->size);
    mul_expr->coeff_sup.resize(mul_expr->size);
    for(size_t i=0; i<expr->size; i++){
        double_interval_mul_expr_coeff(net->ulp, &mul_expr->coeff_inf[i], &mul_expr->coeff_sup[i],
                                        coeff_inf, coeff_sup, 
                                        expr->coeff_inf[i], expr->coeff_sup[i]);
    }
    double_interval_mul_cst_coeff(net->ulp, net->min_denormal, &mul_expr->cst_inf, &mul_expr->cst_sup,
                                    coeff_inf, coeff_sup, expr->cst_inf, expr->cst_sup);
    return mul_expr;

}

Layer_t* get_pred_layer(Network_t* net, Layer_t* curr_layer){
    return curr_layer->pred_layer;
    /*
    if(curr_layer->layer_index == 0){
        return net->input_layer;
    }
    else if(curr_layer->layer_index > 0){
        return net->layer_vec[curr_layer->layer_index - 1];
    }
    else{
        assert(0 && "Pred layer not exist\n");
    }
    */
}

Expr_t* get_mul_expr(Neuron_t* pred_nt, double inf_coff, double supp_coff, bool is_lower){
    Expr_t* mul_expr = NULL;
    if(is_lower){
        if(supp_coff < 0){
            mul_expr =  pred_nt->uexpr;
        }
        else if(inf_coff < 0){
            mul_expr = pred_nt->lexpr;
        }
    }
    else{
        if(supp_coff < 0){
            mul_expr = pred_nt->lexpr;
        }
        else if(inf_coff < 0){
            mul_expr = pred_nt->uexpr;
        }
    }
    return mul_expr;
}

std::vector<double> get_neuron_incomming_weigts(Neuron_t* nt, Layer_t* layer){
    std::vector<double> weights;
    std::vector<size_t> shape =  layer->w_shape;
    weights.reserve(shape[0]);
    auto coll = xt::col(layer->w,nt->neuron_index);
    //std::cout<<"Column of layer, neuron index: ("<<layer->layer_index<<","<<nt->neuron_index<<") is, "<<coll<<std::endl;
    for(size_t i=0; i < shape[0]; i++){
        double coff = coll[i];//layer->w[i,nt->neuron_index];
        weights.push_back(coff);
    }
    return weights;
}

double get_neuron_bias(Neuron_t* nt, Layer_t* layer){
    double cst = layer->b[nt->neuron_index];
    return cst;
}

void create_input_property_vnnlib(Network_t* net, Basic_pre_cond_t* pre_cond){
    for(size_t i=0; i<net->input_dim; i++){
        Neuron_t* nt = net->input_layer->neurons[i];
        nt->lb = -pre_cond->inp_lb[i];
        nt->ub = pre_cond->inp_ub[i];
    }
}

// void create_marked_layer_splitting_constraints(Layer_t* layer){
//     if(layer->is_marked){
//         assert(layer->layer_type == "FC" && "Layer is not marked but creating expression as per marked layer\n");
//         std::vector<size_t> shape =  layer->w_shape;
//         for(auto nt : layer->neurons){
//             if(nt->is_marked){
//                 Expr_t* expr = new Expr_t();
//                 expr->size = shape[0];
//                 expr->coeff_inf.resize(expr->size);
//                 expr->coeff_sup.resize(expr->size);
//                 auto coll = xt::col(layer->w,nt->neuron_index);
//                 for(size_t i=0; i < shape[0]; i++){
//                     double coff = coll[i];
//                     expr->coeff_inf[i] = -coff;
//                     expr->coeff_sup[i] = coff;
//                 }
//                 double cst = layer->b[nt->neuron_index];
//                 expr->cst_inf = -cst;
//                 expr->cst_sup = cst;
                
//                 Constr_t* constr = new Constr_t();
//                 constr->expr = expr;
//                 constr->is_positive = nt->is_active;
//                 layer->constr_vec.push_back(constr);
//             }
//         }
//     }    
// }

// void create_constr_vec_by_size(std::vector<Constr_t*>& constr_vec, std::vector<Constr_t*>& old_vec, size_t constr_size){
//     constr_vec.reserve(old_vec.size());
//     for(size_t i=0; i<old_vec.size(); i++){
//         Constr_t* con = new Constr_t();
//         con->expr = new Expr_t();
//         Constr_t* old_con = old_vec[i];
//         con->expr->size = constr_size;
//         con->expr->cst_inf = old_con->expr->cst_inf;
//         con->expr->cst_sup = old_con->expr->cst_sup;
//         con->expr->coeff_inf.resize(constr_size);
//         con->expr->coeff_sup.resize(constr_size);
//         con->is_positive = old_con->is_positive;
//         constr_vec.push_back(con);
//     }
// }

// void create_constr_vec_with_init_expr(std::vector<Constr_t*>& constr_vec, std::vector<Constr_t*>& old_vec, size_t constr_size){
//     constr_vec.reserve(old_vec.size());
//     for(size_t i=0; i<old_vec.size(); i++){
//         Constr_t* con = declare_constr_t();
//         Constr_t* old_con = old_vec[i];
//         con->expr->size = constr_size;
//         con->expr->cst_inf = 0;
//         con->expr->cst_sup = 0;
//         con->expr->coeff_inf.resize(con->expr->size, 0.0);
//         con->expr->coeff_sup.resize(con->expr->size, 0.0);
//         con->is_positive = old_con->is_positive;
//         constr_vec.push_back(con);
//     }
// }

// void update_independent_constr_relu(Network_t* net, std::vector<Constr_t*>& new_constr_vec, std::vector<Constr_t*>& old_constr_vec,Neuron_t* pred_nt){
//     size_t index = pred_nt->neuron_index;
//     for(size_t i=0; i<old_constr_vec.size(); i++){
//         Constr_t* old_con = old_constr_vec[i];
//         Expr_t* old_con_expr = old_con->expr;
//         Constr_t* new_con = new_constr_vec[i];
//         Expr_t* new_con_expr = new_con->expr;
//         if(old_con_expr->coeff_inf[index] == 0.0 && old_con_expr->coeff_sup[index] == 0.0){
//             new_con_expr->coeff_inf[index] = 0.0;
//             new_con_expr->coeff_sup[index] = 0.0;
//             continue;
//         }

//         Expr_t* mul_expr = get_mul_expr(pred_nt, old_con_expr->coeff_inf[index], old_con_expr->coeff_sup[index], !old_con->is_positive);
//         if(mul_expr != NULL && (old_con_expr->coeff_inf[index] < 0.0 || old_con_expr->coeff_sup[index] < 0.0)){
//             double_interval_mul_expr_coeff(net->ulp,&new_con_expr->coeff_inf[index], &new_con_expr->coeff_sup[index],
//                                             mul_expr->coeff_inf[0], mul_expr->coeff_sup[0],
//                                             old_con_expr->coeff_inf[index], old_con_expr->coeff_sup[index]);
//             double tmp1, tmp2;
//             double_interval_mul_cst_coeff(net->ulp, net->min_denormal, &tmp1, &tmp2,
//                                             mul_expr->cst_inf, mul_expr->cst_sup,
//                                             old_con_expr->coeff_inf[index], old_con_expr->coeff_sup[index]);
//             new_con_expr->cst_inf = new_con_expr->cst_inf + tmp1 + net->min_denormal;
//             new_con_expr->cst_sup = new_con_expr->cst_sup + tmp2 + net->min_denormal;
//         }
//         else{
//             new_con_expr->coeff_inf[index] = 0.0;
//             new_con_expr->coeff_sup[index] = 0.0;
//             double tmp1, tmp2;
//             double_interval_mul_expr_coeff(net->ulp, &tmp1,&tmp2, pred_nt->lb, pred_nt->ub, 
//                                             old_con_expr->coeff_inf[index],old_con_expr->coeff_sup[index]);    
            
//             if(!new_con->is_positive){
// 				new_con_expr->cst_inf = new_con_expr->cst_inf + tmp1;
//                 new_con_expr->cst_sup = new_con_expr->cst_sup - tmp1;
// 			}
// 			else{
//                 new_con_expr->cst_inf = new_con_expr->cst_inf - tmp2;
//                 new_con_expr->cst_sup = new_con_expr->cst_sup + tmp2;
// 			}
//         }
//     }
// }

// void update_dependent_constr_relu(Network_t* net, std::vector<Constr_t*>& new_constr_vec, std::vector<Constr_t*>& old_constr_vec, Expr_t* mul_expr, Neuron_t* pred_nt){
//     size_t index = pred_nt->neuron_index;
//     for(size_t i=0; i<old_constr_vec.size(); i++){
//         Constr_t* old_con = old_constr_vec[i];
//         Expr_t* old_con_expr = old_con->expr;
//         Constr_t* new_con = new_constr_vec[i];
//         Expr_t* new_con_expr = new_con->expr;
//         if(old_con_expr->coeff_inf[index] == 0.0 && old_con_expr->coeff_sup[index] == 0.0){
//             new_con_expr->coeff_inf[index] = 0.0;
//             new_con_expr->coeff_sup[index] = 0.0;
//             continue;
//         }
//         if(mul_expr != NULL && (old_con_expr->coeff_inf[index] < 0.0 || old_con_expr->coeff_sup[index] < 0.0)){
//             double_interval_mul_expr_coeff(net->ulp,&new_con_expr->coeff_inf[index], &new_con_expr->coeff_sup[index],
//                                             mul_expr->coeff_inf[0], mul_expr->coeff_sup[0],
//                                             old_con_expr->coeff_inf[index], old_con_expr->coeff_sup[index]);
//             double tmp1, tmp2;
//             double_interval_mul_cst_coeff(net->ulp, net->min_denormal, &tmp1, &tmp2,
//                                             mul_expr->cst_inf, mul_expr->cst_sup,
//                                             old_con_expr->coeff_inf[index], old_con_expr->coeff_sup[index]);
//             new_con_expr->cst_inf = new_con_expr->cst_inf + tmp1 + net->min_denormal;
//             new_con_expr->cst_sup = new_con_expr->cst_sup + tmp2 + net->min_denormal;
//         }
//         else{
//             new_con_expr->coeff_inf[index] = 0.0;
//             new_con_expr->coeff_sup[index] = 0.0;
//             double tmp1, tmp2;
//             double_interval_mul_expr_coeff(net->ulp, &tmp1,&tmp2, pred_nt->lb, pred_nt->ub, 
//                                             old_con_expr->coeff_inf[index],old_con_expr->coeff_sup[index]);    
            
//             if(!old_con->is_positive){
// 				new_con_expr->cst_inf = new_con_expr->cst_inf + tmp1;
//                 new_con_expr->cst_sup = new_con_expr->cst_sup - tmp1;
// 			}
// 			else{
//                 new_con_expr->cst_inf = new_con_expr->cst_inf - tmp2;
//                 new_con_expr->cst_sup = new_con_expr->cst_sup + tmp2;
// 			}
//         }
//     }
// }


// void update_independent_constr_FC(Network_t* net, std::vector<Constr_t*>& new_constr_vec, std::vector<Constr_t*>& old_constr_vec,Neuron_t* pred_nt){
//     size_t index = pred_nt->neuron_index;
//     for(size_t i =0; i<old_constr_vec.size(); i++){
//         Constr_t* old_con = old_constr_vec[i];
//         Expr_t* old_con_expr = old_con->expr;
//         Constr_t* new_con = new_constr_vec[i];
//         Expr_t* new_con_expr = new_con->expr;
//         if(old_con_expr->coeff_inf[index] == 0.0 && old_con_expr->coeff_sup[index] == 0.0){
//             continue; //new_con_expr already initialized with 0.0
//         }

//         Expr_t* mul_expr = get_mul_expr(pred_nt, old_con_expr->coeff_inf[index], old_con_expr->coeff_sup[index], !old_con->is_positive);
//         if(mul_expr != NULL && (old_con_expr->coeff_inf[index] < 0 || old_con_expr->coeff_sup[index] < 0)){
//             Expr_t* temp_expr = multiply_expr_with_coeff(net, mul_expr, old_con_expr->coeff_inf[index], old_con_expr->coeff_sup[index]);
//             add_expr(net, new_con_expr, temp_expr);
//             delete temp_expr;    
//         }
//         else{
//             double temp1, temp2;
// 			double_interval_mul_cst_coeff(net->ulp, net->min_denormal,&temp1,&temp2,pred_nt->lb,pred_nt->ub,old_con_expr->coeff_inf[index],old_con_expr->coeff_sup[index]);
// 			if(!old_con->is_positive){
// 				new_con_expr->cst_inf = new_con_expr->cst_inf + temp1;
// 				new_con_expr->cst_sup = new_con_expr->cst_sup - temp1;
// 			}
// 			else{
// 				new_con_expr->cst_inf = new_con_expr->cst_inf - temp2;
// 				new_con_expr->cst_sup = new_con_expr->cst_sup + temp2;
// 			}
//         }
//     }
// }

// void update_dependent_constr_FC(Network_t* net, std::vector<Constr_t*>& new_constr_vec, std::vector<Constr_t*>& old_constr_vec, Expr_t* mul_expr, Neuron_t* pred_nt){
//     size_t index = pred_nt->neuron_index;
//     for(size_t i=0; i<old_constr_vec.size(); i++){
//         Constr_t* old_con = old_constr_vec[i];
//         Expr_t* old_con_expr = old_con->expr;
//         Constr_t* new_con = new_constr_vec[i];
//         Expr_t* new_con_expr = new_con->expr;
//         if(old_con_expr->coeff_inf[index] == 0.0 && old_con_expr->coeff_sup[index] == 0.0){
//             continue;
//         }
//         if(mul_expr != NULL && (old_con_expr->coeff_inf[index] < 0.0 || old_con_expr->coeff_sup[index] < 0.0)){
//             Expr_t* temp_expr = multiply_expr_with_coeff(net, mul_expr, old_con_expr->coeff_inf[index], old_con_expr->coeff_sup[index]);
//             add_expr(net, new_con_expr, temp_expr);
//             delete temp_expr; 
//         }
//         else{
//             double temp1, temp2;
// 			double_interval_mul_cst_coeff(net->ulp, net->min_denormal,&temp1,&temp2,pred_nt->lb,pred_nt->ub,old_con_expr->coeff_inf[index],old_con_expr->coeff_sup[index]);
// 			if(!old_con->is_positive){
// 				new_con_expr->cst_inf = new_con_expr->cst_inf + temp1;
// 				new_con_expr->cst_sup = new_con_expr->cst_sup - temp1;
// 			}
// 			else{
// 				new_con_expr->cst_inf = new_con_expr->cst_inf - temp2;
// 				new_con_expr->cst_sup = new_con_expr->cst_sup + temp2;
// 			}
//         }
//     }
// }

// void update_constr_vec_cst(std::vector<Constr_t*> new_constr_vec, std::vector<Constr_t*>& old_constr_vec){
//     for(size_t i=0; i<old_constr_vec.size(); i++){
//         Expr_t* old_con_expr = old_constr_vec[i]->expr;
//         Expr_t* new_con_expr = new_constr_vec[i]->expr;
//         new_con_expr->cst_inf = new_con_expr->cst_inf + old_con_expr->cst_inf;
//         new_con_expr->cst_sup = new_con_expr->cst_sup + old_con_expr->cst_sup;
//     }
// }

// void free_constr_vector_memory(std::vector<Constr_t*>& constr_vec){
//     for(size_t i=0; i<constr_vec.size(); i++){
//         delete constr_vec[i]->expr;
//         delete constr_vec[i];
//     }
//     constr_vec.clear();
// }

void copy_vector_with_negative_vals(std::vector<double> &vec1, std::vector<double> &vec2){
    vec2.reserve(vec1.size());
    for(auto val:vec1){
        vec2.push_back(-val);
    }
}

// Constr_t* declare_constr_t(){
//     Constr_t* con = new Constr_t();
//     con->expr = new Expr_t();
//     return con;
// }

void update_last_layer(Network_t* net){
    size_t out_size = net->output_dim;
    std::vector<std::vector<double>> vec;
    size_t counter = 0;
    for(size_t i=0; i<out_size; i++){
        std::vector<double> v(out_size-1, 0);
        if(i == net->actual_label){
            for(size_t j=0; j<out_size-1; j++){
                v[j] = 1.0;
            }
        }
        else{
            v[counter] = -1.0;
            counter++;
        }
        vec.push_back(v);
    }

    std::vector<double> vec1;
    for(auto v : vec){
        for(auto val : v){
            vec1.push_back(val);
        }
    }

    std::vector<size_t> shape = {out_size, out_size-1};
    xt::xarray<double> aux_layer_w = xt::adapt(vec1,shape);
    // std::cout<<last_layer_w<<std::endl;

    Layer_t* last_layer = net->layer_vec.back();
    xt::xarray<double> new_w = xt::linalg::dot(last_layer->w, aux_layer_w);
    xt::xarray<double> new_b = xt::linalg::dot(last_layer->b, aux_layer_w);
    // std::cout<<last_layer->w<<std::endl;
    // std::cout<<aux_layer_w<<std::endl;
    // std::cout<<last_layer->b<<std::endl;

    // std::cout<<"After multiplication: "<<std::endl;
    // std::cout<<new_w<<std::endl;
    // std::cout<<new_b<<std::endl;

    Layer_t* new_layer = new Layer_t();
    new_layer->is_activation = false;
    new_layer->dims = out_size -1;
    new_layer->layer_index = last_layer->layer_index;
    new_layer->pred_layer = last_layer->pred_layer;
    new_layer->w = new_w;
    new_layer->b = new_b;
    new_layer->w_shape = {last_layer->w_shape[0], out_size-1};
    new_layer->layer_type = "FC";
    net->output_dim = out_size-1;
    
    for(size_t i=0; i<new_layer->dims; i++){
        Neuron_t* nt = new Neuron_t();
        nt->neuron_index = i;
        nt->layer_index = new_layer->layer_index;
        new_layer->neurons.push_back(nt);
    }

    net->layer_vec[net->layer_vec.size()-1] = new_layer;
    delete last_layer;

   
}

