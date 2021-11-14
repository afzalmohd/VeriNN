#include"interval.hh"



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
