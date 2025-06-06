
#ifndef _DEEPPOLY_NETWORK_H_
#define _DEEPPOLY_NETWORK_H_
//#include<z3++.h>
#include<vector>
#include<math.h>
#include<map>
#include <xtensor/xarray.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include "vnnlib.hh"
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
		void print_expr();
		// ~Expr_t(){
		// 	for(size_t i=0; i<constr_vec.size(); i++){
		// 		delete constr_vec[i]->expr;
		// 		delete constr_vec[i];
		// 	}
		// 	constr_vec.clear();
		// }
};

class Neuron_t{
	public:
		size_t neuron_index;
		int layer_index;
		bool is_marked = false;//false means deeppoly's natural encoding
		bool is_active = false; // false means relu is deactivated, true means relu is activated
		double lb=INFINITY;
		double ub=INFINITY;
		double sat_val;
        Expr_t* uexpr;
        Expr_t* lexpr;
		Expr_t* uexpr_b;
		Expr_t* lexpr_b;
		~Neuron_t(){
            delete uexpr;
			uexpr = NULL;
            delete lexpr;
			lexpr = NULL;
			delete uexpr_b;
			uexpr_b = NULL;
			delete lexpr_b;
			lexpr_b = NULL;
			// for(auto constr : constr_vec){
			// 	delete constr->expr;
			// }
		}
		void print_neuron();
};

class Layer_t{
	public:
		size_t dims;
		std::vector<Neuron_t*> neurons;
		Layer_t* pred_layer;
		bool is_activation;
		std::string activation;
        std::string layer_type;
		int layer_index; //input layer consider as -1 indexed
		bool is_marked = false;
		//std::vector<Constr_t*> constr_vec;
		std::vector<size_t> w_shape;
		xt::xarray<double> w;
        xt::xarray<double> b;
		xt::xarray<double> res;

		~Layer_t(){
			for(size_t i=0; i< neurons.size(); i++){
				delete neurons[i];
			}
			neurons.clear();
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
        size_t actual_label;
		size_t pred_label;
		size_t dim_under_analysis;
		std::vector<size_t> verified_out_dims;
		std::vector<double> stds;
		std::vector<double> means;
		size_t counter_class_dim;
		double min_denormal = ldexpl(1.0,-1074);
    	double ulp = ldexpl(1.0,-52);
		VnnLib_t* vnn_lib;
		std::map<size_t, std::vector<size_t>> index_map_dims_to_split;
		std::map<size_t, double> index_vs_err;
		std::vector<size_t> dims_to_split;
		size_t number_of_marked_neurons=0;
		size_t number_of_refine_iter=0;
		

		~Network_t(){
			for(size_t i=0; i<layer_vec.size(); i++){
				delete layer_vec[i];
			}
			layer_vec.clear();
            delete input_layer;
		}

		void print_network();
		void forward_propgate_one_layer(size_t layer_index, xt::xarray<double> &inp);
		void forward_propgate_network(size_t layer_index, xt::xarray<double> &inp);
};

void analyse(Network_t* net, std::string &image_path);
void reset_network(Network_t* net);
void reset_layer(Layer_t* layer);
void mark_layer_and_neurons(Layer_t* layer);
double round_off(double num, size_t prec);

#endif