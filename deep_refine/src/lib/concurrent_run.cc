#include "concurrent_run.hh"
#include <pthread.h>
#include <bits/stdc++.h>
#include "drefine_driver.hh"
#include "milp_refine.hh"
#include "../../deeppoly/optimizer.hh"
#include "milp_mark.hh"
#include "parallel_fns.hh"
pthread_mutex_t lck;
// pthread_mutex_t lck_model;
volatile sig_atomic_t terminate_flag = 0;
int i = 0;
long long size;
bool is_refine = false;
pthread_t thread_id[NUM_THREADS];
std::vector<int> refine_comb;
bool verif_result = true;
std::vector<std::vector<int>> combs;
Network_t *net1 = new Network_t();
std::vector<bool> return_models;
int next_marked_index = 0;
int relu_constr_mine(Layer_t *layer, GRBModel &model, std::vector<GRBVar> &var_vector, size_t var_counter, std::vector<int> &activations, int marked_index,std::vector<Neuron_t*> new_list_mn)
{   
    if(terminate_flag==1){
        pthread_exit(NULL);
    }
    for (size_t i = 0; i < layer->dims; i++)
    {
        Neuron_t *nt = layer->neurons[i];
        Neuron_t *pred_nt = layer->pred_layer->neurons[i];
        if (pred_nt->lb <= 0)
        {
            GRBLinExpr grb_expr = var_vector[var_counter + i] - var_vector[var_counter + i - layer->pred_layer->dims];
            model.addConstr(grb_expr, GRB_EQUAL, 0);
        }
        else if (pred_nt->ub <= 0)
        {
            GRBLinExpr grb_expr = var_vector[var_counter + i];
            model.addConstr(grb_expr, GRB_EQUAL, 0);
        }
        else if (pred_nt->is_marked)
        {   size_t b;
            for(b=0;b<new_list_mn.size();b++){
                if(pred_nt==new_list_mn[b]){
                    break;
                }
            }
            if (activations[b] == 1)
            {
                GRBLinExpr grb_expr = var_vector[var_counter + i] - var_vector[var_counter + i - layer->pred_layer->dims];
                model.addConstr(grb_expr, GRB_EQUAL, 0);
                grb_expr = var_vector[var_counter + i - layer->pred_layer->dims];
                model.addConstr(grb_expr, GRB_GREATER_EQUAL, 0);
            }
            else
            {
                GRBLinExpr grb_expr = var_vector[var_counter + i];
                model.addConstr(grb_expr, GRB_EQUAL, 0);
                grb_expr = var_vector[var_counter + i - layer->pred_layer->dims];
                model.addConstr(grb_expr, GRB_LESS_EQUAL, 0);
            }
        }
        else
        {
            double lb = -pred_nt->lb;
            GRBLinExpr grb_expr = -pred_nt->ub * var_vector[var_counter + i - layer->pred_layer->dims] + pred_nt->ub * lb;
            grb_expr += (nt->ub - lb) * var_vector[var_counter + i];
            model.addConstr(grb_expr, GRB_LESS_EQUAL, 0);

            GRBLinExpr grb_expr1 = var_vector[var_counter + i] - var_vector[var_counter + i - layer->pred_layer->dims];
            model.addConstr(grb_expr1, GRB_GREATER_EQUAL, 0);
        }
    }
    return marked_index;
}

bool add_constraint(Network_t *net, GRBModel &model, std::vector<GRBVar> &var_vector, std::vector<int> &activations, int index,std::vector<Neuron_t*> new_list_mn)
{   if(terminate_flag==1){
        pthread_exit(NULL);
    }
    creating_vars_milp(net, model, var_vector); // populate var_vector with neuron names like x12,x23
    // std::cout<<"add_constraint"<<std::endl;
    size_t var_counter = net->input_layer->dims; // since input layer is not included in net->layer_vec
    next_marked_index = 0;
    for (auto layer : net->layer_vec)
    {
        if (layer->is_activation)
        {
            // std::cout<<"relu called with index "<<index<<std::endl;
            next_marked_index=relu_constr_mine(layer, model, var_vector, var_counter, activations,next_marked_index,new_list_mn);
            // create_relu_constr_milp_refine(layer, model, var_vector, var_counter);
        }
        else
        {
            // std::cout<<"milp_constr_FC_without_marked before"<<std::endl;
            create_milp_constr_FC_without_marked(layer, model, var_vector, var_counter);
            // std::cout<<"milp_constr_FC_without_marked after"<<std::endl;
        }

        var_counter += layer->dims;
    }
    if(Configuration_deeppoly::is_softmax_conf_ce){
        return is_image_verified_softmax_concurrent(net, model, var_vector, activations);
    }
    for (size_t i = 0; i < net->output_dim && verif_result != false; i++)
    // for (size_t i = 0; i < net->output_dim; i++)
    {
        // std::cout<<"inside loop i= "<<i<< std::endl;
        if (i != net->actual_label)
        {
            bool is_already_verified = false;
            for (size_t val : net->verified_out_dims)
            {
                // std::cout<<"val= "<<val<<std::endl;
                if (val == i)
                {

                    // std::cout<<"is_already_verified"<<std::endl;
                    is_already_verified = true;
                }
            }
            if (!is_already_verified)
            {
                // std::cout<<"verif will be called"<<std::endl;
                if (!verify_by_milp_mine(net, model, var_vector, i, true, activations))
                {
                    // std::cout<<"in verify_by_milp_mine with false \n";//<<std::this_thread::get_id()<<std::endl;

                    net->counter_class_dim = i;
                    // return_models[index]=false;
                    verif_result=false;
                    return false;
                    break;
                }
                // else{
                //     net->verified_out_dims.push_back(i);
                // }
            }
            // std::cout<<"b"<<std::endl;
        }
        // std::cout<<"a"<<std::endl;
    }
    return true;
}

bool model_gen(std::vector<int> &activations, int index, Network_t *net,std::vector<Neuron_t*> new_list_mn)
{
    if(terminate_flag==1){
        pthread_exit(NULL);
    }
    GRBModel model = create_grb_env_and_model();
    std::vector<GRBVar> var_vector;
    return add_constraint(net, model, var_vector, activations, index,new_list_mn);
}

//--------------------------------------------------------------------------
int rec_dep=1;
std::vector<int> generateBinaryOutput(int m, int n) {
    std::vector<int> binaryOutput(n, 0);
    
    for (int i = n - 1; i >= 0 && m > 0; --i) {
        binaryOutput[i] = m & 1;
        m >>= 1;
    }
    
    return binaryOutput;
}
void *multi_thread(void *p)
{  
    // std::cout << "thread running is ----- " << std::this_thread::get_id() << std::endl;
    while (verif_result != false)
    {   
        if(terminate_flag==1){
            pthread_exit(NULL);
        }
        pthread_mutex_lock(&lck);
        if (i >= size)
        {
            pthread_mutex_unlock(&lck);
            // std::cout << "exiting " << std::this_thread::get_id() << std::endl;
            break;
        }
        int index = i++;
        pthread_mutex_unlock(&lck);
        // std::cout<<index <<"is being run by "<<std::this_thread::get_id()<<std::endl;
        std::vector<int> result = generateBinaryOutput(index, Global_vars::new_marked_nts.size());
        model_gen(result, index,net1,Global_vars::new_marked_nts);
        // break;
    }
    pthread_exit(NULL);
}


bool looper(Network_t *net){
    net1=net;
    Configuration_deeppoly::grb_num_thread = 1;
    while(1){
        size = std::pow(2, Global_vars::new_marked_nts.size());
        i=0;
        // std::cout<<"size = "<<size<<" i = "<<i<<std::endl; 
        for(int i=0;i<NUM_THREADS;i++){
            // pthread_t thread_id;
            // std::cout<<"create "<<thread_id[i]<<std::endl;
            int threadCreateResult =pthread_create(&thread_id[i], NULL, multi_thread, NULL);
            if (threadCreateResult != 0) {
                std::cerr << "Error creating thread: " << strerror(threadCreateResult) << std::endl;
                return 1;
            }
        }
        for(int i = 0; i < NUM_THREADS; i++){
                // std::cout<<"JOIN\n";
                pthread_join(thread_id[i], NULL);
                // std::cout<<"JOIN_after\n";
        }
        // std::cout<<"after join loop\n";
        if(verif_result==false){return false;}
        // std::cout<<"after verif result if\n";
        if(is_refine==true)
        {
            if(Global_vars::new_marked_nts.size()<7)
            {
                run_milp_mark_with_milp_refine_mine(net);
                Global_vars::iter_counts++;
                is_refine=false;
            }
            else{
                Configuration_deeppoly::is_concurrent = false;
                Configuration_deeppoly::grb_num_thread = NUM_GUROBI_THREAD;
                // Configuration_deeppoly::is_reset_marked_nts = true;
                return run_cegar_milp_mark_milp_refine(net);
            }

        }
        else{return true;}
        // std::cout<<"after else\n";
    }
    return true;
}
bool rec_con(Network_t *net, std::vector<int> prev_comb, int new_mn)
{   
    Configuration_deeppoly::grb_num_thread = 1;
    std::cout<<std::endl;
    std::vector<std::vector<int>> new_combs;
    bool retval =looper(net);
    if(retval==true){
        return true;
    }
    else{
        return false;
    }
}
