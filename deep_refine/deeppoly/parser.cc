#include "parser.hh"
#include "deeppoly_configuration.hh"
#include<fstream>


void init_network(Network_t* net, std::string &filepath){
    std::fstream newfile;
    newfile.open(filepath, std::ios::in);
    if(newfile.is_open()){
        std::string tp;
        std::string relu_str = "ReLU";
        std::string matmul_str = "Gemm";
        size_t layer_index = 0;
        while (getline(newfile, tp)){
            if(tp != ""){
                if(tp == relu_str){
                    getline(newfile, tp);
                    Layer_t* layer = create_layer(false, "", "FC");
                    parse_string_to_xarray(layer, tp, false); //weight matrix
                    getline(newfile, tp);
                    parse_string_to_xarray(layer, tp, true); //bias matrix
                    
                    layer->layer_index = layer_index;
                    create_neurons_update_layer(layer);
                    net->layer_vec.push_back(layer);
                    layer_index++;
                    
                    Layer_t* relu_layer = create_layer(true, "ReLU", "FC"); //Creating relu layer seperately
                    relu_layer->dims = layer->dims;
                    relu_layer->layer_index = layer_index;
                    create_neurons_update_layer(relu_layer);
                    net->layer_vec.push_back(relu_layer);
                    layer_index++;
                }
                else if(tp == matmul_str){
                    getline(newfile, tp);
                    Layer_t* layer = create_layer(false, "", "FC");
                    parse_string_to_xarray(layer, tp, false); //weight matrix
                    getline(newfile, tp);
                    parse_string_to_xarray(layer, tp, true); //bias matrix
                    
                    layer->layer_index = layer_index;
                    create_neurons_update_layer(layer);
                    net->layer_vec.push_back(layer);
                    layer_index++;
                }
            }
        }
        net->input_layer = create_input_layer(Configuration_deeppoly::input_dim);
        net->numlayers = net->layer_vec.size();
        net->input_dim = net->input_layer->dims;
        net->output_dim = net->layer_vec.back()->dims;
        pred_layer_linking(net);
    }
}

void pred_layer_linking(Network_t* net){
    for(size_t i=0; i < net->layer_vec.size(); i++){
        Layer_t* layer = net->layer_vec[i];
        if(i == 0){
            layer->pred_layer = net->input_layer;
        }
        else{
            layer->pred_layer = net->layer_vec[i-1];
        }
    }
}

void parse_string_to_xarray(Layer_t* layer, std::string weights, bool is_bias){
    std::vector<double> weight_vec;
    char comma = ',';
    char left_brac = '[';
    char right_brac = ']';
    std::string acc = "";
    size_t rows = 0;
    size_t cols = 0;
    size_t count_cols = 0;
    size_t count_right_brac = 0;
    for(size_t i=0; i<weights.size();i++){
        if(weights[i] == comma){
            if(acc != ""){
                double val = std::stod(acc);
                weight_vec.push_back(val);
                acc = "";   
                count_cols++; 
                count_right_brac = 0;
            }    
        }
        else if(weights[i] == right_brac){
            if(acc != ""){
                double val = std::stod(acc);
                weight_vec.push_back(val);
                acc = "";
                count_cols++; 
                cols = count_cols;   
            } 
            count_right_brac++;
            if(count_right_brac == 1){
                count_cols = 0;
                rows++;
            }

        }
        else if(weights[i] != left_brac && !std::isspace(weights[i])){
            acc += weights[i];
        }
    }
    if(acc != ""){
         double val = std::stod(acc);
         weight_vec.push_back(val);
    }

    if(is_bias){
        std::vector<size_t> shape = {cols};
        layer->b = xt::adapt(weight_vec, shape);
        auto aisehi = layer->b.shape();
        std::cout<<"Bias matrix shape: " <<xt::adapt(layer->b.shape())<<std::endl;
        layer->dims = cols;
    }
    else{
        std::vector<size_t> shape = {rows, cols};
        xt::xarray<double> temp = xt::adapt(weight_vec, shape);
        layer->w = xt::transpose(temp);
        layer->w_shape = {cols, rows};
        std::cout<<"Weight matrix shape: "<<xt::adapt(layer->w.shape())<<std::endl;
    }
}

void parse_input_image(Network_t* net, std::string &image_path, size_t image_index){
    std::fstream newfile;
    newfile.open(image_path, std::ios::in);
    if(newfile.is_open()){
        std::string tp;
        size_t image_counter = 0;
        while (getline(newfile, tp)){
            if(tp != ""){
                if(image_counter == image_index){
                    parse_image_string_to_xarray_one(net,tp);
                    break;
                }
                else{
                    image_counter++;
                }
            }
        }
    }
}

void parse_image_string_to_xarray_one(Network_t* net, std::string &image_str){
    std::vector<double> vec;
    std::string acc = "";
    size_t pixel_counter = 0;
    bool is_first = true;
    for(size_t i=0; i<image_str.size();i++){
        if(image_str[i] == IMAGE_DELIMETER){
            if(acc != ""){
                if(is_first){
                    net->actual_label = std::stoi(acc);
                    is_first = false;
                    acc = "";
                }
                else{
                    double val = std::stod(acc);
                    vec.push_back(val);
                    acc = "";
                    pixel_counter++;
                }
            }
        }
        else if(!std::isspace(image_str[i])){
            acc += image_str[i];
        }
    }
    if(acc != ""){
        double val = std::stod(acc);
        vec.push_back(val);
        pixel_counter++;
    }
    assert(pixel_counter == net->input_dim && "Pixel count mismatch");
    std::vector<size_t> shape = {net->input_dim};
    net->input_layer->res = xt::adapt(vec,shape) / 255;
}

Layer_t* create_layer(bool is_activation, std::string activation, std::string layer_type){
    Layer_t* layer = new Layer_t();
    layer->is_activation = is_activation;
    if(is_activation){
        layer->activation = activation;
    }
    else{
        layer->layer_type = layer_type;
    }
    return layer;
}

void create_neurons_update_layer(Layer_t* layer){
    for(size_t i=0; i < layer->dims;i++){
        Neuron_t* nt = new Neuron_t();
        nt->neuron_index = i;
        nt->layer_index = layer->layer_index;
        layer->neurons.push_back(nt);
    }
}

Layer_t* create_input_layer(size_t dim){
    Layer_t* layer = new Layer_t();
    layer->dims = dim;
    layer->layer_index = -1;
    create_neurons_update_layer(layer); 
    return layer;
}



// VnnLib_t* parse_vnnlib_file(std::string& prop_file){
//     VnnLib_t* verinn_lib  = new VnnLib_t();
//     std::fstream vnn_file;
//     vnn_file.open(prop_file, std::ios::in);
//     std::string white_space = "[ \t\r\n\f]*";
//     std::string comparison_str = "()";
//     if(vnn_file.is_open()){
//         std::string tp;
//         size_t max_index_in_vars = 0;
//         size_t max_index_out_vars = 0;
//         size_t num_in_vars = 0;
//         size_t num_out_vars = 0;
//         std::vector<double> inp_lb;
//         std::vector<double> inp_ub;
//         std::vector<double> out_lb;
//         std::vector<double> out_ub;
//         bool is_set_vec_size = false;
//         bool is_constr_visited = false;
//         std::string reg_var_str = ".*declare-const"+white_space+"([a-zA-Z]+)_([0-9]+)"+white_space+"([a-zA-Z]+).*";
//         std::string reg_constr_str = ".*assert.*(<=|>=|<|>)"+white_space+"([a-zA-Z]+)_([0-9]+)"+white_space+"([-]*[0-9]*\.[0-9]*).*";
//         std::regex reg_for_vars(reg_var_str);
//         std::regex reg_for_constr(reg_constr_str);
//         while (getline(vnn_file, tp)){
//             if(tp != ""){
//                 std::cmatch m_var;
//                 bool is_dec = regex_search(tp.c_str(), m_var, reg_for_vars);
//                 if(is_dec){
//                     get_vars(m_var, max_index_in_vars, max_index_out_vars, num_in_vars, num_out_vars);
//                     assert(!is_constr_visited && "Constrains declares before variable declaration in vnnlib property file");
//                 }
//                 else{
//                     std::cmatch m_constr;
//                     bool is_constr = regex_search(tp.c_str(), m_constr, reg_for_constr);
//                     if(is_constr){
//                         is_constr_visited = true;
//                         if(!is_set_vec_size){
//                             init_bound_vecs(max_index_in_vars, max_index_out_vars, inp_lb, inp_ub, out_lb, out_ub);
//                             is_set_vec_size = true;
//                         }
//                         parse_constraints_vnnlib(m_constr, inp_lb, inp_ub, out_lb, out_ub);
//                     }
//                 }

//             }
//         }
//         std::cout<<"Num input vars: "<<num_in_vars<<" , max index input vars: "<<max_index_in_vars<<std::endl;
//         std::cout<<"Num out vars: "<<num_out_vars<<" , max index out vars: "<<max_index_out_vars<<std::endl;
//         std::cout<<"Input bounds: "<<std::endl;
//         for(size_t i=0; i<inp_lb.size(); i++){
//             std::cout<<"lb: "<<inp_lb[i]<<", ub: "<<inp_ub[i]<<std::endl;
//         }
//         std::cout<<"Output bounds: "<<std::endl;
//         for(size_t i=0; i<out_lb.size(); i++){
//             std::cout<<"lb: "<<out_lb[i]<<", ub: "<<out_ub[i]<<std::endl;
//         }
//         Configuration_deeppoly::input_dim = num_in_vars;
//         verinn_lib->input_dims = num_in_vars;
//         verinn_lib->output_dims = num_out_vars;
//         verinn_lib->inp_lb = inp_lb;
//         verinn_lib->inp_ub = inp_ub;
//         verinn_lib->out_lb = out_lb;
//         verinn_lib->out_ub = out_ub;
//     }
//     else{
//         assert(0 && "vnnlib file could not open");
//     }

//     return verinn_lib;
// }

void init_bound_vecs(size_t max_inp_index, size_t max_out_index, std::vector<double>& inp_lb, std::vector<double>& inp_ub, std::vector<double>& out_lb, std::vector<double>& out_ub){
    inp_lb.reserve(max_inp_index+1);
    inp_ub.reserve(max_out_index+1);
    for(size_t i=0; i<max_inp_index+1; i++){
        inp_lb.push_back(-INFINITY);
        inp_ub.push_back(INFINITY);
    }
    for(size_t i=0; i<max_out_index+1; i++){
        out_lb.push_back(-INFINITY);
        out_ub.push_back(INFINITY);
    }
    
}

// void get_vars(std::cmatch& m_var, size_t& max_index_in_vars, size_t& max_index_out_vars, size_t& num_in_vars, size_t& num_out_vars){
//     std::string var_name = m_var[1].str();
//     if(var_name[0] == 'X'){
//         size_t var_index = std::stoul(m_var[2]);
//         if(max_index_in_vars < var_index){
//             max_index_in_vars = var_index;
//         }
//         num_in_vars += 1;
//     }
//     else if(var_name[0] == 'Y'){
//         size_t var_index = std::stoul(m_var[2]);
//         if(max_index_out_vars < var_index){
//             max_index_out_vars = var_index;
//         }
//         num_out_vars += 1;
//     }
//     else{
//          assert(0 && "Unknown variable in vnnlib property file");
//     } 
// }

void parse_constraints_vnnlib(std::cmatch& m_var, std::vector<double>& in_lb, std::vector<double>& in_ub, std::vector<double>& out_lb, std::vector<double>& out_ub){
    
    std::string op = m_var[1].str();
    std::string var_str = m_var[2].str();
    std::string var_index_str = m_var[3].str();
    size_t var_index = std::stoul(var_index_str);
    std::string bound_str = m_var[4].str();
    double bound = std::stod(bound_str);
    // std::cout<<op<<" , "<<var_str<<" , "<<var_index_str<<" , "<<bound_str<<std::endl;
    // std::cout<<op<<" , "<<var_str<<" , "<<var_index<<" , "<<bound<<std::endl;
    if(op == "<=" || op == "<"){
        if(var_str[0] == 'X'){
            in_ub[var_index] = bound;
        }
        else if(var_str[0] == 'Y'){
            out_ub[var_index] = bound;
        }
        else{
            std::cout<<"Var name: "<<var_str<<std::endl;
            assert(0 && "Unknown variables in constrains parser vnnlib property file");
        }
    }
    else if(op == ">=" || op == ">"){
        if(var_str[0] == 'X'){
            in_lb[var_index] = bound;
        }
        else if(var_str[0] == 'Y'){
            out_lb[var_index] = bound;
        }
        else{
            std::cout<<"Var name: "<<var_str<<std::endl;
            assert(0 && "Unknown variables in constrains parser vnnlib property file");
        }
    }
    else{
        std::cout<<"Operator: "<<op<<std::endl;
        assert(0 && "Unknown operator in constraints in vnnlib property file");
    }
}