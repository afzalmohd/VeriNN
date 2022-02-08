
#ifndef _DEEPPOLY_NETWORK_H_
#define _DEEPPOLY_NETWORK_H_
#include<z3++.h>
#include<vector>
#include<math.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>

class Float_number{
    public:
        double inf;
        double sup;
};

//class Expr_t;

// class Constr_t{
// 	public:
// 		Expr_t* expr;
// 		bool is_positive;
// 		//size_t layer_index;
// 		//size_t neuron_index;  //Layer and neuron index by which this constraint generated
// 		// ~Constr_t(){
// 		// 	delete expr;
// 		// }
// 		void deep_copy(Constr_t* constr);
// 		void print_constr();
// 		//bool is_same_generator(Constr_t* constr);
// };

class Sparse_neuron_t{
	public:
		size_t neuron_index;
		int layer_index;
		bool is_active;
};

class Expr_t{
	public:
		std::vector<double> coeff_inf;
        std::vector<double> coeff_sup;
		//std::vector<Constr_t*> constr_vec; // Constraints added by other layers neurons in backsubstitutions
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
		double unmarked_lb = INFINITY;
		double unmarked_ub = INFINITY;
		bool is_back_prop_active = false;
		double back_prop_lb;
		double back_prop_ub;
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
		std::vector<std::vector<Sparse_neuron_t*>> IIS;
		std::vector<Neuron_t*> marked_neurons;
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
		std::vector<size_t> verified_out_dims;
		size_t counter_class_dim;
		double min_denormal = ldexpl(1.0,-1074);
    	double ulp = ldexpl(1.0,-52);

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