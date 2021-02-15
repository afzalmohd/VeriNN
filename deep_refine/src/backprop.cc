#include "backprop.hh"
#include "network.hh"


void back_propgate(Network_t* net){
    for(auto layer:net->layer_vec){
        if(layer->layer_index != 0){
            
        }
    }
}


int main(){
    std::string filepath = "/home/u1411251/Documents/Phd/tools/ERAN/tf_verify/fppolyForward.txt";
    Network_t* net = new Network_t();
    z3::context c;
    init_network(c,net,filepath);
    set_predecessor_and_z3_var(c,net);
    init_z3_expr(c,net);
    printf("\nCheck..\n");
    return 0;
}
