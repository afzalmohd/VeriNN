#include "network.hh"
#include "deeppoly_configuration.hh"
#include "parser.hh"
#include "analysis.hh"

int deeppoly_set_params(int argc, char* argv[]){
    Configuration_deeppoly::init_options(argc, argv);
    if(Configuration_deeppoly::vm.count("help")){
        return 1;
    }
    return 0;
}

Network_t* deeppoly_initialize_network(){
    Network_t* net = new Network_t();
    init_network(net, Configuration_deeppoly::net_path);
    return net;
}

void deeppoly_parse_input_image_string(Network_t* net, std::string & image_str){
    parse_image_string_to_xarray_one(net,image_str);
}

void deeppoly_reset_network(Network_t* net){
    reset_network(net);
}

size_t execute_network(Network_t* net){   
    net->forward_propgate_network(0, net->input_layer->res);
    auto pred_label = xt::argmax(net->layer_vec.back()->res);
    return pred_label[0];
}

bool run_deeppoly(Network_t* net){    
    bool is_varified = forward_analysis(net);
    return is_varified;
} 

VnnLib_t* parse_vnnlib(std::string& file_path){
    VnnLib_t* verinn_lib =  parse_vnnlib_file(file_path);
    std::cout<<"Pre condition........."<<std::endl;
    print_pre_cond(verinn_lib->pre_cond_vec, "");
    std::cout<<"Post condition........."<<std::endl;
    print_post_cond(verinn_lib->out_prp, "");
    return verinn_lib;
}

