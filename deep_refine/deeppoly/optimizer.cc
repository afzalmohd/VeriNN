#include "optimizer.hh"
#include "helper.hh"

void compute_bounds_using_gurobi(Network_t* net, Layer_t* layer, Neuron_t* nt, Expr_t* expr, bool is_minimize){
    Layer_t* pred_layer = get_pred_layer(net, layer);
    try{
        GRBEnv env = GRBEnv(true);
        env.start();
        GRBModel model = GRBModel(env);
        std::vector<GRBVar> var_vector(pred_layer->dims);
        //add variables to the model
        for(size_t i=0; i<pred_layer->dims; i++){
            Neuron_t* nt1 = pred_layer->neurons[i];
            std::string var_str = "i_"+std::to_string(i);
            GRBVar x = model.addVar(-nt1->lb, nt->ub, 0.0, GRB_CONTINUOUS, var_str);
            var_vector.push_back(x);
        }

        //add objective function to the model
        GRBLinExpr obj_expr = 0;
        obj_expr.addTerms(&expr->coeff_sup[0], &var_vector[0], var_vector.size());
        if(is_minimize){
            model.setObjective(obj_expr, GRB_MINIMIZE);
        }
        else{
            model.setObjective(obj_expr, GRB_MAXIMIZE);
        }

        for(auto con : nt->constr_vec){
            Expr_t* con_expr = con->expr;
            GRBLinExpr grb_expr = 0;
            grb_expr.addTerms(&con_expr->coeff_sup[0], &var_vector[0], var_vector.size());
            if(con->is_positive){
                model.addConstr(grb_expr, GRB_GREATER_EQUAL, -con_expr->cst_sup);
            }
            else{
                model.addConstr(grb_expr, GRB_LESS_EQUAL, -con_expr->cst_sup); 
            }
        }

        model.optimize();

        if(is_minimize){
            nt->lb = model.get(GRB_DoubleAttr_ObjVal);
        }
        else{
            nt->ub = model.get(GRB_DoubleAttr_ObjVal);
        }

    }
    catch(GRBException e){
        std::cout<<e.getMessage()<<std::endl;
    }
    catch(...){
        std::cout<<"Exception during optimization"<<std::endl;
    }

}