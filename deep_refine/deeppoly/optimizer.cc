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
            GRBVar x = model.addVar(-nt1->lb, nt1->ub, 0.0, GRB_CONTINUOUS, var_str);
            var_vector[i] = x;
        }

        //add objective function to the model
        GRBLinExpr obj_expr = 0;
        if(is_minimize){
            std::vector<double> coeffs;
            copy_vector_with_negative_vals(expr->coeff_inf, coeffs);
            obj_expr.addTerms(&coeffs[0], &var_vector[0], var_vector.size());
            obj_expr -= expr->cst_inf;
        }
        else{
            obj_expr.addTerms(&expr->coeff_sup[0], &var_vector[0], var_vector.size());
            obj_expr += expr->cst_sup;
        }
        
        if(is_minimize){
            model.setObjective(obj_expr, GRB_MINIMIZE);
        }
        else{
            model.setObjective(obj_expr, GRB_MAXIMIZE);
        }

        for(auto con : expr->constr_vec){
            Expr_t* con_expr = con->expr;
            GRBLinExpr grb_expr = 0;
            if(con->is_positive){
                grb_expr.addTerms(&con_expr->coeff_sup[0], &var_vector[0], var_vector.size());
                grb_expr += con_expr->cst_sup;
                model.addConstr(grb_expr, GRB_GREATER_EQUAL, 0);
            }
            else{
                std::vector<double> coeffs;
                copy_vector_with_negative_vals(expr->coeff_inf, coeffs);
                grb_expr.addTerms(&coeffs[0], &var_vector[0], var_vector.size());
                grb_expr -= con_expr->cst_inf; //cst_inf already in negative form
                model.addConstr(grb_expr, GRB_LESS_EQUAL, 0); 
            }
        }

        model.optimize();

        if(is_minimize){
            double tmp = model.get(GRB_DoubleAttr_ObjVal);
            if(tmp > -nt->lb){
                nt->lb = -tmp;
            }
        }
        else{
            double tmp = model.get(GRB_DoubleAttr_ObjVal);
            if(tmp < nt->ub){
                nt->ub = tmp;
            }
        }

    }
    catch(GRBException e){
        std::cout<<e.getMessage()<<std::endl;
    }
    catch(...){
        std::cout<<"Exception during optimization"<<std::endl;
    }

}