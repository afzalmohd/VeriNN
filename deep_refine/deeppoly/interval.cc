#include"interval.hh"

/*
The functions "double_interval_mul", "double_interval_mul_expr_coeff", "double_interval_mul_cst_coeff"
are copied from elina_box_meetjoin.c which is a file of ELINA library.
The source file is available here https://github.com/eth-sri/ELINA/blob/master/elina_zonotope/elina_box_meetjoin.c

*/


void double_interval_mul(double *a_inf, double *a_sup, double b_inf, double b_sup, double c_inf, double c_sup){
	if(c_inf<=0){
		/* interval c is positive */
		if(b_inf<=0){
			/*interval b is positive*/
			if((b_inf==0) || (c_inf==0)){
				*a_inf = 0.0;
			}
			else{
				*a_inf = b_inf * -c_inf;
			}
			if((b_sup==0) || (c_sup==0)){
				*a_sup = 0.0;
			}
			else{
				*a_sup = b_sup * c_sup;
			}
		}
		else if(b_sup<=0){
			/* interval b is negative */
			if((c_sup==0) || (b_inf==0)){
				*a_inf = 0.0;
			}
			else{
			 	*a_inf = c_sup*b_inf;
			}
			if((c_inf==0) || (b_sup==0)){
				*a_sup = 0.0;
			}
			else{
			 	*a_sup = -c_inf*b_sup;
			}
		}
		else{
			/* there is 0 in between for b */
			if((c_sup==0) || (b_inf==0)){
				*a_inf = 0.0;
			}
			else{
				*a_inf = b_inf * c_sup;
			}
			if((c_sup==0) || (b_sup==0)){
				*a_sup = 0.0;
			}
			else{
				*a_sup = b_sup * c_sup;
			}
		}
	}
	else if(c_sup<=0){
		/* interval c is negative */
		if(b_inf<=0){
			/*interval b is positive*/
			if((b_sup==0) || (c_inf==0)){
				*a_inf = 0.0;
			}
			else{
				*a_inf = b_sup*c_inf;
			}
			if((b_inf==0) || (c_sup==0)){
				*a_sup = 0.0;
			}
			else{
				*a_sup = -b_inf*c_sup;
			}
		}
		else if(b_sup<=0){
			/* interval b is negative */
			if((b_sup==0) || (c_sup==0)){
				*a_inf = 0.0;
			}
			else{
				*a_inf = b_sup * -c_sup;
			}
			if((b_inf==0) || (c_inf == 0 )){
				*a_sup = 0.0;
			}
			else{
				*a_sup = b_inf * c_inf;
			} 
		}
		else{
			/* there is 0 in between for b */
			if((c_inf==0) || (b_sup==0)){
				*a_inf = 0.0;
			}
			else{
				*a_inf = b_sup*c_inf;
			}
			if((c_inf==0) || (b_inf==0)){
				*a_sup = 0.0;
			}
			else{
				*a_sup = b_inf*c_inf;
			} 
		}
	}
	else if(b_inf<=0){
		/* interval b is positive */
		if(c_inf<=0){
			/*interval c is positive */
			if((b_inf==0) || (c_inf==0)){
				*a_inf = 0.0;
			}
			else{
				*a_inf = -b_inf * c_inf;
			}
			if((b_sup==0) || (c_sup==0)){
				*a_sup = 0.0;
			}
			else{
				*a_sup = b_sup * c_sup;
			}
		}
		else if(c_sup<=0){
			/* interval c is negative */
			if((b_sup==0) || (c_inf == 0)){
				*a_inf = 0.0;
			}
			else{
				*a_inf = b_sup*c_inf;
			}
			if((b_inf==0) || (c_sup==0)){
				*a_sup = 0.0;
			}
			else{
				*a_sup = -b_inf*c_sup;
			}
		}
		else{
			/* there is 0 in between for c */
			if((b_sup==0) || (c_inf==0)){
				*a_inf = 0.0;
			}
			else{
				*a_inf = b_sup * c_inf;
			}
			if((b_sup==0) || (c_sup==0)){
				*a_sup = 0.0;
			}
			else{
				*a_sup = b_sup * c_sup;
			}
		}
	}
	else if(b_sup<=0){
		/* interval b is negative */
		if(c_inf <= 0){
			/* interval c is positive */
			if((b_inf==0) || (c_sup==0)){
				*a_inf = 0.0;
			}
			else{
				*a_inf = b_inf * c_sup;
			}
			if((b_sup==0) || (c_inf==0)){
				*a_sup = 0.0;
			}
			else{
				*a_sup = b_sup * -c_inf;
			}
		}
		else if(c_sup<=0){
			/* interval c is negative */
			if((b_sup==0) || (c_sup==0)){
				*a_inf = 0.0;
			}
			else{
				*a_inf = -b_sup * c_sup;
			}
			if((b_inf==0) || (c_inf==0)){
				*a_sup = 0.0;
			}
			else{
				*a_sup = b_inf * c_inf;
			} 
		}
		else{
			/* there is 0 in between for c */
			if((b_inf == 0) || (c_sup==0)){
				*a_inf = 0.0;
			}
			else{
				*a_inf = b_inf * c_sup;
			}
			if((b_inf==0) || (c_inf==0)){
				*a_sup = 0.0;
			}
			else{
				*a_sup = b_inf * c_inf;
			}
		}
	}
	else{
		/* there is 0 in between for both b and c */
		double tmp_inf1 = b_sup*c_inf;
		double tmp_sup1 = b_inf*c_inf;
		double tmp_inf2 = b_inf*c_sup;
		double tmp_sup2 = b_sup*c_sup;
		*a_inf = fmax(tmp_inf1, tmp_inf2);
		*a_sup = fmax(tmp_sup1, tmp_sup2);
	}
}

void double_interval_mul_expr_coeff(double fac, double * res_inf, double *res_sup, double inf, double sup, double inf_expr, double sup_expr){
	double_interval_mul(res_inf,res_sup,inf,sup,inf_expr,sup_expr);
	double maxA = fmax(fabs(inf_expr),fabs(sup_expr));
	double tmp1, tmp2;
	double_interval_mul(&tmp1,&tmp2, inf, sup, maxA*fac, maxA*fac);
	*res_inf += tmp1;
	*res_sup += tmp2;
}

void double_interval_mul_cst_coeff(double fac, double min_denormal, double * res_inf, double *res_sup, double inf, double sup, double inf_expr, double sup_expr){
	double_interval_mul_expr_coeff(fac, res_inf, res_sup, inf, sup, inf_expr, sup_expr);
	*res_inf += min_denormal;
	*res_sup += min_denormal;	
}
