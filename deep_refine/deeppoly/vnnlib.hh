#ifndef _MY_VNN_LIB_
#define _MY_VNN_LIB_

#include<vector>
#include<string>

class Vnnlib_prp_t{
    std::string type; //basic/rel/conj/disj
    std::vector<std::string> basic_prp;
    std::vector<Vnnlib_prp_t*> comp_prp;
};

class VnnLib_t{
	public:
		size_t input_dims;
		size_t output_dims;
		std::vector<double> inp_lb;
		std::vector<double> inp_ub;
		std::vector<double> out_lb;
		std::vector<double> out_ub;
        Vnnlib_prp_t* out_prp;
};

#endif