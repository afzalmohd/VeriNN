#ifndef _MY_VNN_LIB_
#define _MY_VNN_LIB_

#include<vector>
#include<string>
#include<regex>
#include<math.h>

class Basic_post_cond_t{
    public:
        std::string type; //basic/rel
        std::string op;
        std::string lhs;
        std::string rhs;
};

class Basic_pre_cond_t{
    public:
        std::vector<double> inp_lb;
		std::vector<double> inp_ub;

};

class Vnnlib_post_cond_t{
    public:
        std::string type; //basic/rel/conj/disj
        std::vector<Basic_post_cond_t*> basic_prp;
        std::vector<Vnnlib_post_cond_t*> comp_prp;
        std::vector<Vnnlib_post_cond_t*> verified_sub_prp;
};

class VnnLib_t{
	public:
		size_t input_dims;
		size_t output_dims;
        size_t pre_cond_count = 0;
        bool is_conj_pre_cond = false;
        bool is_disj_pre_cond = false;
        std::vector<Basic_post_cond_t*> direct_assert_prps;
        Vnnlib_post_cond_t* out_prp;
        std::vector<Basic_pre_cond_t*> pre_cond_vec; // all preconditions are in disjunctive form
};

class Parse_vnnlib_prp_t{
    std::string white_space_corner = "[ \t\r\n\f]*";
    std::string white_space = "[ \t\r\n\f]+";
    std::string token_str = "[a-zA-Z0-9\\.\\-_]+";
    std::string comparison_suffix = white_space_corner+token_str+white_space+token_str+white_space_corner;
    std::string geq_str = white_space_corner+"\\("+white_space_corner+">="+comparison_suffix+"\\)"+white_space_corner;
    std::string ge_str = white_space_corner+"\\("+white_space_corner+">"+comparison_suffix+"\\)"+white_space_corner;
    std::string leq_str = white_space_corner+"\\("+white_space_corner+"<="+comparison_suffix+"\\)"+white_space_corner;
    std::string le_str = white_space_corner+"\\("+white_space_corner+"<"+comparison_suffix+"\\)"+white_space_corner;
    public:
        std::string rel_str_tokenize = "(<=|>=|<|>)"+white_space_corner+"("+token_str+")"+white_space+"("+token_str+")";
        std::string rel_str = "("+geq_str+"|"+ge_str+"|"+leq_str+"|"+le_str+")+";
        std::string conj_str = white_space_corner+"\\("+white_space_corner+"and"+rel_str+"\\)";
        std::string disj_str = white_space_corner+"\\("+white_space_corner+"or("+conj_str+")+\\)";
        std::string assrt_str = white_space_corner+"\\("+white_space_corner+"assert"+rel_str+"\\)";
        std::string var_str = ".*declare-const"+white_space+"([a-zA-Z]+)_([0-9]+)"+white_space+"([a-zA-Z]+).*";

        void get_direct_constraints(VnnLib_t* vnn_obj, std::string& line_str);
        void extract_prp_comb(VnnLib_t* vnnlib, Vnnlib_post_cond_t* prp, std::fstream& vnn_file, std::string& line_str, std::string search_type);
        void extract_prp_conj(VnnLib_t* vnn_lib, Vnnlib_post_cond_t* prp, std::string& line_str, std::string& search_string);
        void get_tokens(std::string& str, std::string& search_str);
};

VnnLib_t* parse_vnnlib_file(std::string& file_path);
void get_vars(std::cmatch& m_var, size_t& max_index_in_vars, size_t& max_index_out_vars, size_t& num_in_vars, size_t& num_out_vars);
void init_bound_vecs(size_t max_inp_index,std::vector<double>& inp_lb, std::vector<double>& inp_ub);
bool is_number(std::string& str);
size_t get_var_index(std::string var_str, bool is_input_var);
void set_basic_pre_cond(VnnLib_t* vnn_obj, std::string lhs, std::string rhs, std::string op, bool is_lhs_num);
void set_basic_post_cond(VnnLib_t* vnn_obj, std::string lhs, std::string rhs, std::string op);
void set_post_cond_comb(Basic_post_cond_t* prp, std::string lhs, std::string rhs, std::string op);
void print_post_cond(Vnnlib_post_cond_t* post_cond, std::string ident);
void print_basic_post_cond(Basic_post_cond_t* cond, std::string ident);
void print_pre_cond(std::vector<Basic_pre_cond_t*>& pre_cond_vec, std::string ident);
void print_pre_cond_basic(Basic_pre_cond_t* pre_cond, std::string ident);

#endif