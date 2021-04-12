//Main lives here
#include "include/drefine_api.h"

int main(int argc, char* argv[]){

  // cmd line processing code here
  // std::string path = "/home/u1411251/Documents/Phd/tools/VeriNN/deep_refine/";
  
  // bool is_my_test = false;

  // std::string path = "";
  // std::string filepath = "";
  // std::string net_path = "";

  // if(is_my_test){
  //   filepath = path+"benchmarks/fppolyForward_my_test.txt"; // output of eran
  //   net_path = path+"benchmarks/mnist_my_text.tf"; //small test case
  // }
  // else{
  //   filepath = path+"benchmarks/fppolyForward.txt"; // output of eran
  //   net_path = path+"benchmarks/mnist_relu_3_50.tf"; // network to be verified
  // }
  // std::string dataset_path = path+"benchmarks/mnist_test.csv"; // image file
  find_refine_nodes(argc, argv);
  return 0;
}

//   initialize_drefine();
//   markings = run_drefine();
//   // print markings
//   return 0;
// }
