//Main lives here
#include "include/drefine_api.h"

int main(int argc, char* argv[]){

  //find_refine_nodes(argc, argv); //argv are the arguments passes like abstraction output, neural network file,
                               // image dataset file, epsilon etc. 
  run_refine_poly(argc, argv);
  printf("\nCheck..main\n");
  return 0;
}
