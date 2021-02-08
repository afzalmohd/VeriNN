#include <z3++.h>
#include <map>
#include <vector>

typedef struct expr_t{
	double *coeff;
	double cst;
	size_t * dim;
    size_t size;
}expr_t;

typedef struct neuron_t{
	std::string lb; //datatype string since some bounds are "inf"
	std::string ub;
	std::vector<double> lcoeffs;
	std::vector<double> ucoeffs;
}neuron_t;

typedef struct layer_t{
	size_t dims;
	std::vector<neuron_t*> neurons;
	bool is_activation;
	std::string activation;
    size_t layer_index;
}layer_t;


typedef struct fppoly_t{
    std::vector<layer_t*> layer_map;
	std::vector<z3::expr> expr_map;
    size_t numlayers = 0;//Other than input layer
    layer_t * input_layer;
    z3::expr input_expr;	
}fppoly_t;
