
#ifndef _DEEPPOLY_NETWORK_H_
#define _DEEPPOLY_NETWORK_H_
#include<z3++.h>
#include<vector>
#include<math.h>
#include "configuration.hh"

#include <xtensor/xarray.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>

class Float_number{
    public:
        double inf;
        double sup;
};

class Expr_t{
	public:
		std::vector<double> coeff_inf;
        std::vector<double> coeff_sup;
		double cst_inf;
        double cst_sup;
		size_t size=0;
		void deep_copy(Expr_t* expr);
};

class Neuron_t{
	public:
		size_t neuron_index;
		double lb=INFINITY;
		double ub=INFINITY;
        Expr_t* uexpr;
        Expr_t* lexpr;
		Expr_t* uexpr_b;
		Expr_t* lexpr_b;
		//std::vector<Neuron_t*> pred_neurons;
		~Neuron_t(){
			//for(size_t i=0; i<pred_neurons.size(); i++){
			//	delete pred_neurons[i];
			//}
            delete uexpr;
            delete lexpr;
			delete uexpr_b;
			delete lexpr_b;
		}
		void print_neuron();
};

class Layer_t{
	public:
		size_t dims;
		std::vector<Neuron_t*> neurons;
		bool is_activation;
		std::string activation;
        std::string layer_type;
		int layer_index; //input layer consider as -1 indexed
		std::vector<size_t> w_shape;
		xt::xarray<double> w;
        xt::xarray<double> b;
		xt::xarray<double> res;

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
        Layer_t* input_layer;
		size_t numlayers = 0;//Other than input layer
		size_t input_dim = 0;
		size_t output_dim=0;
		double epsilon = 0;
        int actual_label;
		double min_denormal = ldexpl(1.0,-1074);
    	double ulp = ldexpl(1.0,-52);

		~Network_t(){
			for(size_t i=0; i<layer_vec.size(); i++){
				delete layer_vec[i];
			}
            delete input_layer;
		}

		void print_network();
		void forward_propgate_one_layer(size_t layer_index, xt::xarray<double> &inp);
		void forward_propgate_network(size_t layer_index, xt::xarray<double> &inp);
};

void execute_neural_network(Network_t* net, std::string &image_path);

#endif