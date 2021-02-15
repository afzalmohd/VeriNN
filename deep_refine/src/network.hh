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
		size_t neuron_index;
		std::string lb; //datatype string since some bounds are "inf"
		std::string ub;
		std::vector<double> lcoeffs;
		std::vector<double> ucoeffs;
		std::vector<Neuron_t*> pred_neurons;
		z3::context c;
		z3::expr nt_z3_var = c.bool_val(true);
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
		z3::context c;
		z3::expr layer_expr = c.bool_val(true);
};

class Network_t{
	public:
		std::vector<Layer_t*> layer_vec;
		std::vector<z3::expr> expr_vec;
		size_t numlayers = 0;//Other than input layer
		Layer_t* input_layer;
		size_t input_dim=2;//has to be change as per the input
		Network_t();
};

std::vector<std::string> parse_string(std::string ft);
void init_expr_coeffs(Neuron_t& nt, std::vector<std::string> &coeffs, bool is_upper);
void init_network(z3::context &c, Network_t* net, std::string file_path);
void set_predecessor_layer_activation(z3::context& c, Layer_t* layer, Layer_t* prev_layer);
void set_predecessor_layer_matmul(z3::context& c, Layer_t* layer, Layer_t* prev_layer);
void set_predecessor_and_z3_var(z3::context &c, Network_t* net);
z3::expr get_expr_from_double(z3::context &c, double item);
void init_z3_expr_neuron(z3::context &c, Neuron_t* nt);
void init_z3_expr_layer(z3::context &c, Layer_t* layer);
void init_z3_expr(z3::context &c, Network_t* net);