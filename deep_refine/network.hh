#include <z3++.h>
#include <map>
#include <vector>

class Expr_t{
	public:
		double *coeff;
		double cst;
		size_t * dim;
		size_t size;
};

class Neuron_t{
	public:
		std::string lb; //datatype string since some bounds are "inf"
		std::string ub;
		std::vector<double> lcoeffs;
		std::vector<double> ucoeffs;
		z3::context c;
		z3::expr z_lexpr = c.bool_val(true);
		z3::expr z_uexpr = c.bool_val(true);
};

class Layer_t{
	public:
		size_t dims;
		std::vector<Neuron_t*> neurons;
		std::vector<z3::expr> vars;
		bool is_activation;
		std::string activation;
		size_t layer_index;
};

class Fppoly_t{
	public:
		std::vector<Layer_t*> layer_vec;
		std::vector<z3::expr> expr_vec;
		size_t numlayers = 0;//Other than input layer
		Layer_t input_layer;
		Fppoly_t();
};

std::vector<std::string> parse_string(std::string ft);
void add_expr(Neuron_t& nt, std::vector<std::string> &coeffs, bool is_upper);
void init_network(z3::context &c, Fppoly_t& fp, std::string file_path);