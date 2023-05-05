#include "vnnlib.hh"
#include<fstream>
#include<cassert>
#include<iostream>

VnnLib_t* parse_vnnlib_file(std::string& file_path){
    VnnLib_t* vnn_lib = new VnnLib_t();
    Vnnlib_post_cond_t* prp = new Vnnlib_post_cond_t();
    vnn_lib->out_prp = prp;
    std::fstream vnn_file;
    vnn_file.open(file_path, std::ios::in);
    Parse_vnnlib_prp_t* parse_vnnlib_obj = new Parse_vnnlib_prp_t();
    if(vnn_file.is_open()){
        std::string tp;
        size_t max_index_in_vars = 0;
        size_t max_index_out_vars = 0;
        size_t num_in_vars = 0;
        size_t num_out_vars = 0;
        bool is_set_vec_size = false;
        bool is_constr_visited = false;
        while(getline(vnn_file, tp)){
            if(tp != ""){
                std::cmatch m_var;
                bool is_dec = std::regex_search(tp.c_str(), m_var, std::regex(parse_vnnlib_obj->var_str));
                if(is_dec){
                    get_vars(m_var, max_index_in_vars, max_index_out_vars, num_in_vars, num_out_vars);
                    assert(!is_constr_visited && "Constrains declares before variable declaration in vnnlib property file");
                }
                else{
                    if(tp.find("assert") != std::string::npos){
                        is_constr_visited = true;
                        if(!is_set_vec_size){
                            vnn_lib->input_dims = num_in_vars;
                            vnn_lib->output_dims = num_out_vars;
                            is_set_vec_size = true;
                        }
                        auto is_only_assert = std::regex_search(tp.c_str(), m_var, std::regex(parse_vnnlib_obj->assrt_str));
                        if(is_only_assert){
                            parse_vnnlib_obj->get_direct_constraints(vnn_lib, tp);
                        }
                        else{
                            parse_vnnlib_obj->extract_prp_comb(vnn_lib, prp, vnn_file, tp, "disj");
                        }
                    }
                }

            }
        }
        // std::cout<<"Max index input var: "<<max_index_in_vars<<" , Max index output var: "<<max_index_out_vars<<std::endl;
        // std::cout<<"Number of input vars: "<<num_in_vars<<" , Number of output vars: "<<num_out_vars<<std::endl;
        assert(max_index_in_vars+1 == num_in_vars && "Something wrong with input variables indexing");
        assert(max_index_out_vars+1 == num_out_vars && "Something wrong with output vars indexing");
    }
    else{
        std::cout<<file_path<<std::endl;
        assert(0 && "something wrong with vnnlib file path");
    }
    
    if(vnn_lib->direct_assert_prps.size() > 0 && vnn_lib->out_prp->comp_prp.size() > 0){
        assert(0 && "Define output properties in two ways");
    }
    if(vnn_lib->direct_assert_prps.size() > 0){
        vnn_lib->out_prp->type = "conj";
        vnn_lib->out_prp->basic_prp = vnn_lib->direct_assert_prps;
        vnn_lib->direct_assert_prps.clear();
    }

    return vnn_lib;
}

void Parse_vnnlib_prp_t::get_direct_constraints(VnnLib_t* vnn_obj, std::string& line_str){
    std::cmatch m;
    auto b = std::regex_search(line_str.c_str(), m, std::regex(this->rel_str_tokenize));
    if(b){
        std::string op = m[1].str();
        std::string lhs = m[2].str();
        std::string rhs = m[3].str();
        if(is_number(lhs) && is_number(rhs)){
            std::cout<<op<<" "<<lhs<<" "<<rhs<<std::endl;
            assert(0 && "wrong constraint pattern");
        }
        else if(is_number(lhs)){
            if(rhs[0] == 'X'){
                if(!vnn_obj->is_conj_pre_cond){
                    vnn_obj->is_conj_pre_cond = true;
                    Basic_pre_cond_t* basic_pre_cond = new Basic_pre_cond_t();
                    vnn_obj->pre_cond_vec.push_back(basic_pre_cond);
                    init_bound_vecs(vnn_obj->input_dims, basic_pre_cond->inp_lb, basic_pre_cond->inp_ub);
                }
                set_basic_pre_cond(vnn_obj, lhs, rhs, op, true);
            }
            else if(rhs[0] == 'Y'){
               set_basic_post_cond(vnn_obj, lhs, rhs, op); 
            }
            else{
                std::cout<<rhs<<std::endl;
                assert(0 && "wrong variable name");
            }
        }
        else if(is_number(rhs)){
            if(lhs[0] == 'X'){
                if(!vnn_obj->is_conj_pre_cond){
                    vnn_obj->is_conj_pre_cond = true;
                    Basic_pre_cond_t* basic_pre_cond = new Basic_pre_cond_t();
                    vnn_obj->pre_cond_vec.push_back(basic_pre_cond);
                    init_bound_vecs(vnn_obj->input_dims, basic_pre_cond->inp_lb, basic_pre_cond->inp_ub);
                }
                set_basic_pre_cond(vnn_obj, lhs, rhs, op, false);
            }
            else if(lhs[0] == 'Y'){
                set_basic_post_cond(vnn_obj, lhs, rhs, op);
            }
            else{
                std::cout<<rhs<<std::endl;
                assert(0 && "wrong variable name");
            }
        }
        else{
            if(lhs[0] == 'Y' && rhs[0] == 'Y'){
                Basic_post_cond_t* basic_prp = new Basic_post_cond_t();
                basic_prp->type = "rel";
                basic_prp->op = op;
                basic_prp->lhs = lhs;
                basic_prp->rhs = rhs;
                vnn_obj->direct_assert_prps.push_back(basic_prp);
            }
            else{
                std::cout<<op<<" "<<lhs<<" "<<rhs<<std::endl;
            }
        }
    }
    else{
        std::cout<<line_str<<std::endl;
        assert(0 && "Wrong constraint");
    }
}

size_t get_var_index(std::string var_str, bool is_input_var){
    std::cmatch var_index;
    if(is_input_var){
        auto b = std::regex_search(var_str.c_str(), var_index, std::regex("X_([0-9]+)"));
        if(b){
            size_t index = std::stoul(var_index[1].str());
            return index;
        }
    }
    else{
        auto b = std::regex_search(var_str.c_str(), var_index, std::regex("Y_([0-9]+)"));
        if(b){
            size_t index = std::stoul(var_index[1].str());
            return index;
        }
    }

    std::cout<<var_str<<std::endl;
    assert(0 && "wrong variable indexing");
    return 0;
}

void set_basic_pre_cond(VnnLib_t* vnn_obj, std::string lhs, std::string rhs, std::string op, bool is_lhs_num){
    if(is_lhs_num){
        size_t index = get_var_index(rhs, true);
        Basic_pre_cond_t* pre_cond = vnn_obj->pre_cond_vec.back();
        if(op == ">=" || op == ">"){
            pre_cond->inp_ub[index] = std::stod(lhs);
        }
        else if(op == "<=" || op == "<"){
            pre_cond->inp_lb[index] = std::stod(lhs);
        }
        else{
            std::cout<<op<<" "<<lhs<<" "<<rhs<<std::endl;
            assert(0 && "wrong operator");
        }
       
    }
    else{
        size_t index = get_var_index(lhs, true);
        Basic_pre_cond_t* pre_cond = vnn_obj->pre_cond_vec.back();
        if(op == ">=" || op == ">"){
            pre_cond->inp_lb[index] = std::stod(rhs);
        }
        else if(op == "<=" || op == "<"){
            pre_cond->inp_ub[index] = std::stod(rhs);
        }
        else{
            std::cout<<op<<" "<<lhs<<" "<<rhs<<std::endl;
            assert(0 && "wrong operator");
        }
    }
}

void set_basic_post_cond(VnnLib_t* vnn_obj, std::string lhs, std::string rhs, std::string op){
    Basic_post_cond_t* basic_prp = new Basic_post_cond_t();
    basic_prp->type = "basic";
    basic_prp->op = op;
    basic_prp->lhs = lhs;
    basic_prp->rhs = rhs;
    vnn_obj->direct_assert_prps.push_back(basic_prp);
}

void Parse_vnnlib_prp_t::extract_prp_conj(VnnLib_t* vnn_lib, Vnnlib_post_cond_t* prp, std::string& line_str, std::string& search_string){
    std::cmatch m;
    while(1){
        auto b = std::regex_search(line_str.c_str(), m, std::regex(search_string));
        if(!b){
            break;
        }
        //std::cout<<m[0]<<std::endl;
        std::string op = m[1].str();
        std::string lhs = m[2].str();
        std::string rhs = m[3].str();
        if(lhs[0] == 'Y' || rhs[0] == 'Y'){
            Basic_post_cond_t* basic_prp = new Basic_post_cond_t();
            set_post_cond_comb(basic_prp, lhs, rhs, op);
            prp->basic_prp.push_back(basic_prp);
        }
        else if(lhs[0] == 'X' || rhs[0] == 'X'){
            if(is_number(lhs)){
                set_basic_pre_cond(vnn_lib, lhs, rhs, op, true);
            }
            else if(is_number(rhs)){
                set_basic_pre_cond(vnn_lib, lhs, rhs, op, false);
            }
            else{
                std::cout<<m[0]<<std::endl;
                assert(0 && "Something wrong with constraints");
            }
            
        }
        else{
            std::cout<<m[0]<<std::endl;
            assert(0 && "Unknown variable");
        }
        line_str = m.suffix();
    }
}

void set_post_cond_comb(Basic_post_cond_t* prp, std::string lhs, std::string rhs, std::string op){
    if(is_number(lhs) || is_number(rhs)){
        prp->type = "basic";
    }
    else{
        prp->type = "rel";
    }
    prp->lhs = lhs;
    prp->rhs = rhs;
    prp->op = op;
}

void Parse_vnnlib_prp_t::extract_prp_comb(VnnLib_t* vnnlib, Vnnlib_post_cond_t* prp, std::fstream& vnn_file, std::string& line_str, std::string search_type){
    std::cmatch m;
    if(search_type == "disj"){
        auto b = std::regex_search(line_str.c_str(), m, std::regex(this->disj_str));
        std::string or_str = "(or";
        if(b){
            std::string match_str = m[0].str();
            //std::cout<<match_str<<std::endl;
            while (1){   
                b = std::regex_search(match_str.c_str(), m, std::regex(this->conj_str));
                if(!b){
                    break;
                }
                //std::cout<< m[0]<<std::endl;
                std::string m_str = m[0].str();
                bool is_input_cond = m_str.find("X_") != std::string::npos;
                bool is_output_cond = m_str.find("Y_") != std::string::npos;
                if(is_input_cond && !is_output_cond){
                    Basic_pre_cond_t* pre_cond_basic = new Basic_pre_cond_t();
                    init_bound_vecs(vnnlib->input_dims, pre_cond_basic->inp_lb, pre_cond_basic->inp_ub);
                    vnnlib->pre_cond_vec.push_back(pre_cond_basic);
                    extract_prp_conj(vnnlib, prp, m_str, this->rel_str_tokenize);  
                }
                else if(!is_input_cond && is_output_cond){
                    prp->type = "disj";
                    Vnnlib_post_cond_t* prp_conj = new Vnnlib_post_cond_t();
                    prp_conj->type = "conj";
                    extract_prp_conj(vnnlib, prp_conj, m_str, this->rel_str_tokenize);
                    prp->comp_prp.push_back(prp_conj);
                }
                else{
                    std::cout<<m_str<<std::endl;
                    assert(0 && "something wrong in output property");
                }
                
                match_str = m.suffix();
            }
        }
        else if(line_str.find(or_str) != std::string::npos){
            std::string tp;
            while(1){
                if(getline(vnn_file, tp)){
                    if(tp == "))"){
                        break;
                    }
                    else{
                        bool is_input_cond = tp.find("X_") != std::string::npos;
                        bool is_output_cond = tp.find("Y_") != std::string::npos;
                        if(is_input_cond && !is_output_cond){
                            Basic_pre_cond_t* pre_cond_basic = new Basic_pre_cond_t();
                            init_bound_vecs(vnnlib->input_dims, pre_cond_basic->inp_lb, pre_cond_basic->inp_ub);
                            vnnlib->pre_cond_vec.push_back(pre_cond_basic);
                            extract_prp_conj(vnnlib, prp, tp, this->rel_str_tokenize);  
                        }
                        else if(!is_input_cond && is_output_cond){
                            prp->type = "disj";
                            Vnnlib_post_cond_t* prp_conj = new Vnnlib_post_cond_t();
                            prp_conj->type = "conj";
                            extract_prp_conj(vnnlib, prp_conj, tp, this->rel_str_tokenize);
                            prp->comp_prp.push_back(prp_conj);
                        }
                        else{
                            std::cout<<tp<<std::endl;
                            assert(0 && "something wrong in output property");
                        }
                    }

                }
                else{
                    assert(0 && "something wrong in output property");
                }
            }
        }
        else{
            this->extract_prp_comb(vnnlib, prp, vnn_file, line_str, "conj");
        }
    }
    else if(search_type == "conj"){
        auto b = std::regex_search(line_str.c_str(), m, std::regex(this->conj_str));
        if(b){
            bool is_input_cond = line_str.find("X_") != std::string::npos;
            bool is_output_cond = line_str.find("Y_") != std::string::npos;
            if(is_input_cond && !is_output_cond){
                Basic_pre_cond_t* pre_cond_basic = new Basic_pre_cond_t();
                init_bound_vecs(vnnlib->input_dims, pre_cond_basic->inp_lb, pre_cond_basic->inp_ub);
                vnnlib->pre_cond_vec.push_back(pre_cond_basic);
                extract_prp_conj(vnnlib, prp, line_str, this->rel_str_tokenize); 
            }
            else if(!is_input_cond && is_output_cond){
                prp->type = "conj";
                this->extract_prp_conj(vnnlib, prp, line_str, this->rel_str_tokenize);
            }
            else{
                std::cout<<line_str<<std::endl;
                assert(0 && "something wrong");
            }
        }
        else{
            std::cout<<"Constrain: "<<line_str<<std::endl;
            assert(0 && "Not valid conjuctive constraints");
        }
    }
}


void get_vars(std::cmatch& m_var, size_t& max_index_in_vars, size_t& max_index_out_vars, size_t& num_in_vars, size_t& num_out_vars){
    std::string var_name = m_var[1].str();
    if(var_name[0] == 'X'){
        size_t var_index = std::stoul(m_var[2]);
        if(max_index_in_vars < var_index){
            max_index_in_vars = var_index;
        }
        num_in_vars += 1;
    }
    else if(var_name[0] == 'Y'){
        size_t var_index = std::stoul(m_var[2]);
        if(max_index_out_vars < var_index){
            max_index_out_vars = var_index;
        }
        num_out_vars += 1;
    }
    else{
         assert(0 && "Unknown variable in vnnlib property file");
    } 
}

void init_bound_vecs(size_t vec_size, std::vector<double>& inp_lb, std::vector<double>& inp_ub){
    inp_lb.reserve(vec_size);
    inp_ub.reserve(vec_size);
    for(size_t i=0; i<vec_size; i++){
        inp_lb.push_back(-INFINITY);
        inp_ub.push_back(INFINITY);
    }
}

bool is_number(std::string& s){
    std::string::const_iterator it = s.begin();
    char minus = '-';
    char dot = '.';
    char plus = '+';
    while (it != s.end() && (std::isdigit(*it) || *it == minus || *it == dot || *it == plus)) ++it;
    return !s.empty() && it == s.end();
}

void print_post_cond(Vnnlib_post_cond_t* post_cond, std::string ident){
    if(post_cond->type == "disj"){
        std::cout<<ident<<"or"<<std::endl;
        ident = ident+" ";
        for(Basic_post_cond_t* cond : post_cond->basic_prp){
            print_basic_post_cond(cond, ident);
        }
        for(Vnnlib_post_cond_t* comp_cond:post_cond->comp_prp){
            print_post_cond(comp_cond, ident);
        }
    }
    else if(post_cond->type == "conj"){
        std::cout<<ident<<"and"<<std::endl;
        ident = ident+" ";
        for(Basic_post_cond_t* cond : post_cond->basic_prp){
            print_basic_post_cond(cond, ident);
        }
        for(Vnnlib_post_cond_t* comp_cond:post_cond->comp_prp){
            print_post_cond(comp_cond, ident);
        }
    }
    else{
        std::cout<<post_cond->type<<std::endl;
        assert(0 && "unknown property type");
    }
}

void print_basic_post_cond(Basic_post_cond_t* cond, std::string ident){
    std::cout<<ident<<"("<<cond->lhs<<" "<<cond->op<<" "<<cond->rhs<<")"<<std::endl;
}

void print_pre_cond(std::vector<Basic_pre_cond_t*>& pre_cond_vec, std::string ident){
    if(pre_cond_vec.size() > 1){
        std::cout<<ident<<"or"<<std::endl;
        ident += " ";
    }
    for(Basic_pre_cond_t* pre_cond : pre_cond_vec){
        std::cout<<ident<<"and"<<std::endl;
        print_pre_cond_basic(pre_cond, ident);
    }
}

void print_pre_cond_basic(Basic_pre_cond_t* pre_cond, std::string ident){
    ident += " ";
    for(size_t i=0; i<pre_cond->inp_lb.size(); i++){
        std::cout<<ident<<"("<<pre_cond->inp_lb[i]<<" <= X_"<<i<<" <= "<<pre_cond->inp_ub[i]<<")"<<std::endl;
    }
}
