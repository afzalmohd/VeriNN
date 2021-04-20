#ifndef _MY_NETWORK_H_
#define _MY_NETWORK_H_
#include <z3++.h>
#include <map>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include "configuration.hh"

class Expr_t{
	public:
		double *coeff;
		double cst;
		size_t * dim;
		size_t size;

		~Expr_t(){
			delete coeff;
			delete dim;
		}

};

class Neuron_t{
	public:
		size_t neuron_index;
		double lb=NAN; //datatype string since some bounds are "inf"
		double ub=NAN;
		std::vector<double> lcoeffs;
		std::vector<double> ucoeffs;
		std::vector<Neuron_t*> pred_neurons;
		z3::context& c;
		z3::expr nt_z3_var = c.bool_val(true);
		z3::expr z_lexpr = c.bool_val(true);
		z3::expr z_uexpr = c.bool_val(true);
		z3::expr affine_expr = c.bool_val(true);
		Neuron_t(z3::context &c):c(c){
		}
		~Neuron_t(){
			for(size_t i=0; i<pred_neurons.size(); i++){
				delete pred_neurons[i];
			}
		}
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
		std::tuple<size_t,size_t,size_t> w_shape;
		xt::xarray<double> w;
        xt::xarray<double> b;
		xt::xarray<double> res;
		z3::context& c;
		z3::expr c_expr = c.bool_val(true);
		z3::expr b_expr = c.bool_val(true);
		z3::expr merged_expr = c.bool_val(true);
		z3::expr layer_prop = c.bool_val(true);
		Layer_t(z3::context& c): c(c){}

		~Layer_t(){
			for(size_t i=0; i< neurons.size(); i++){
				delete neurons[i];
			}
		}


		void print_layer();
};

class Network_t{
	public:
		std::vector<Layer_t*> layer_vec;
		std::vector<z3::expr> expr_vec;
		std::vector<std::tuple<size_t,size_t,size_t>> wt_shapes;
		size_t numlayers = 0;//Other than input layer
		Layer_t* input_layer;
		size_t input_dim = Configuration::input_dim;
		size_t output_dim=0;
		double epsilon = Configuration::epsilon;
		bool is_my_test = false;
		xt::xarray<double> im;
		xt::xarray<double> candidate_ce;

		z3::context& c;
		z3::expr prop_expr = c.bool_val(true);
		z3::expr back_subs_prop = c.bool_val(true);
		Network_t(z3::context& c): c(c){
			std::cout<<"Network Constructor called"<<std::endl;
		}

		~Network_t(){
			delete input_layer;
			for(size_t i=0; i<layer_vec.size(); i++){
				delete layer_vec[i];
			}
		}

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
void create_prop(z3::context &c, Network_t* net);
void init_input_box(z3::context &c, Network_t* net);
z3::expr get_expr_from_double(z3::context &c, double item);
bool is_number(std::string s);


#endif