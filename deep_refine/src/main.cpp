//Main lives here
#include "include/drefine_api.h"

int main(){

  // cmd line processing code here
  // std::string path = "/home/u1411251/Documents/Phd/tools/VeriNN/deep_refine/";
  std::string path = "";
  std::string filepath = path+"benchmarks/fppolyForward.txt"; // output of eran
  //std::string net_path = path+"benchmarks/mnist_relu_3_50.tf"; // neural network
  std::string net_path = path+"benchmarks/mnist_my_text.tf"; //small test case
  std::string dataset_path = path+"benchmarks/mnist_test.csv"; // image file
  find_refine_nodes(filepath, net_path, dataset_path);
  return 0;
}

//   initialize_drefine();
//   markings = run_drefine();
//   // print markings
//   return 0;
// }
