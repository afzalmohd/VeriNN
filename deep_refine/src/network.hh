#include <z3++.h>
#include <map>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>

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

		void print_neuron();
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
		std::tuple<size_t,size_t,size_t> w_shape;
		xt::xarray<double> w;
        xt::xarray<double> b;
		xt::xarray<double> res;

		void print_layer();
};

class Network_t{
	public:
		std::vector<Layer_t*> layer_vec;
		std::vector<z3::expr> expr_vec;
		std::vector<std::tuple<size_t,size_t,size_t>> wt_shapes;
		size_t numlayers = 0;//Other than input layer
		Layer_t* input_layer;
		size_t input_dim=2;//has to be change as per the input
		size_t output_dim=0;
		xt::xarray<double> im;
		Network_t();
		void print_network();
		void forward_propgate_one_layer(size_t layer_index, xt::xarray<double> &inp);
		void forward_propgate_network(size_t layer_index, xt::xarray<double> &inp);
};

void init_expr_coeffs(Neuron_t& nt, std::vector<std::string> &coeffs, bool is_upper);
void init_network(z3::context &c, Network_t* net, std::string file_path);
void set_weight_dims(Network_t* net);
void init_net_weights(Network_t* net, std::string &filepath);
void init_images_pixels(Network_t* net, std::string &image_path);
std::vector<std::string> parse_string(std::string ft);
void parse_string_to_xarray(Network_t* net, std::string weights, bool is_bias, size_t layer_index);
void parse_image_string_to_xarray(Network_t* net, std::string &image_path);
void parse_image_string_to_xarray_one(Network_t* net, std::string &image_str);
