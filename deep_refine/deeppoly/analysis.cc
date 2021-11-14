#include "analysis.hh"
#include "interval.hh"

void create_input_layer_expr(Network_t* net){
    Layer_t* layer = net->input_layer;
    for(size_t i=0; i < layer->dims; i++){
        Neuron_t* nt = layer->neurons[i];
        nt->ub = layer->res[i] + net->epsilon;
        nt->lb = layer->res[i] - net->epsilon;
        if(nt->ub > 1.0){
            nt->ub = 1.0;
        }
        if(nt->lb < 0.0){
            nt->lb = 0.0;
        }
        nt->uexpr = new Expr_t();
        nt->uexpr->cst_inf = nt->ub;
        nt->uexpr->cst_sup = nt->ub;
        nt->uexpr->size = 1;
        
        nt->lexpr = new Expr_t();
        nt->lexpr->cst_inf = -nt->lb;
        nt->lexpr->cst_sup = -nt->lb;
        nt->lexpr->size = 1;
    }
}

void forward_layer_FC(Network_t* net, int layer_index){
    Layer_t* curr_layer = net->layer_vec[layer_index];
    Layer_t* prev_layer;
    if(layer_index == 0){
        prev_layer = net->input_layer;
    }
    else{
        prev_layer = net->layer_vec[layer_index-1];
    }

}

void create_neuron_expr_FC(Neuron_t* nt, Layer_t* layer){
    std::vector<size_t> shape =  layer->w_shape;
    nt->uexpr = new Expr_t();
    nt->uexpr_b = new Expr_t();
    nt->lexpr_b = new Expr_t();
    nt->uexpr->size = shape[0];
    nt->uexpr_b->size = shape[0];
    nt->lexpr_b->size = shape[0];
    //nt->lexpr->size = shape[0];
    for(int i=0; i < shape[0]; i++){
        double coff = layer->w[i,nt->neuron_index];
        nt->uexpr->coeff_inf.push_back(-coff);
        nt->uexpr->coeff_sup.push_back(coff);
        nt->uexpr_b->coeff_inf.push_back(-coff);
        nt->uexpr_b->coeff_sup.push_back(coff);
        nt->lexpr_b->coeff_inf.push_back(-coff);
        nt->lexpr_b->coeff_sup.push_back(coff);
    }
    double cst = layer->b[nt->neuron_index];
    nt->uexpr->cst_inf = -cst;
    nt->uexpr->cst_sup = cst;
    nt->uexpr_b->cst_inf = -cst;
    nt->uexpr_b->cst_sup = cst;
    nt->lexpr_b->cst_inf = -cst;
    nt->lexpr_b->cst_sup = cst;
    nt->lexpr = nt->uexpr;
}

void update_neuron_bound_back_substitution(Network_t* net, int layer_index, Neuron_t* nt){
    Layer_t* pred_layer;
    if(layer_index == 0){
        pred_layer = net->input_layer;
        nt->lb = fmin(nt->lb, compute_lb_from_expr(pred_layer, nt->lexpr_b));
        nt->ub = fmin(nt->ub, compute_ub_from_expr(pred_layer, nt->uexpr_b));
    }
    else{
        pred_layer = net->layer_vec[layer_index-1];
        nt->lb = fmin(nt->lb, compute_lb_from_expr(pred_layer, nt->lexpr_b));
        nt->ub = fmin(nt->ub, compute_ub_from_expr(pred_layer, nt->uexpr_b));


    }
}

void update_neuron_lexpr_b(Network_t* net, Layer_t* pred_layer, Neuron_t* nt){
    if(pred_layer->is_activation){

    }
    else if(pred_layer->layer_type == "FC"){

    }
}


double compute_lb_from_expr(Layer_t* pred_layer, Expr_t* expr){
    double res = expr->cst_inf;
    for(size_t i=0; i < expr->size-1; i++){
        double temp1, temp2;
        double_interval_mul(&temp1, &temp2, expr->coeff_inf[i], expr->coeff_sup[i], 
                                pred_layer->neurons[i]->lb, pred_layer->neurons[i]->ub);
        res += temp1;
    }
    return res;
}

double compute_ub_from_expr(Layer_t* pred_layer, Expr_t* expr){
    double res = expr->cst_sup;
    for(size_t i=0; i < expr->size-1; i++){
        double temp1, temp2;
        double_interval_mul(&temp1, &temp2, expr->coeff_inf[i], expr->coeff_sup[i], 
                                pred_layer->neurons[i]->lb, pred_layer->neurons[i]->ub);
        res += temp2;
    }
    return res;
}

