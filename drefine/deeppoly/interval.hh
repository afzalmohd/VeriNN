#ifndef _DEEPPOLY_INTERVAL_H_
#define _DEEPPOLY_INTERVAL_H_
#include<cmath>
void double_interval_mul(double *a_inf, double *a_sup, double b_inf, double b_sup, double c_inf, double c_sup);
void double_interval_mul_expr_coeff(double fac, double * res_inf, double *res_sup, double inf, double sup, double inf_expr, double sup_expr);
void double_interval_mul_cst_coeff(double fac, double min_denormal, double * res_inf, double *res_sup, double inf, double sup, double inf_expr, double sup_expr);

#endif