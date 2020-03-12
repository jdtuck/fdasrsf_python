#define _USE_MATH_DEFINES // for C
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include "misc_funcs.h"

void oclogit_warp_grad(int *n1, int *T1, double *alpha, double *nu, double *q, int *y, int *max_itri, double *toli, double *deltaOi, double *deltagi, int *displayi, double *gamout, double *Oout){

	/* dereference inputs
	alpha, beta, q should be normalized by norm */
	int TT = *T1;  // number of sample points
	int n = *n1;  // number of dimension of curves R^n
	int max_itr = *max_itri;
	double tol = *toli, deltaO = *deltaOi, deltag = *deltagi;
	double display = *displayi;

	// Looping and temp variables
	int j, k, l, jj;
	int p = 20;
	int p1 = 10;
	int itr = 0;
	double binsize1 = 1.0;
	double *t = malloc(sizeof(double) * TT);
	double *gam1 = malloc(sizeof(double) * TT);
	double *ones = malloc(sizeof(double) * TT);
	double *f_basis = malloc(sizeof(double) * TT*p);
	double *q_tilde = malloc(sizeof(double) * TT*n);
	double *q_tmp = malloc(sizeof(double) * TT*n);
	double *q_tilde_diff = malloc(sizeof(double) * TT*n);
	double *cbar = malloc(sizeof(double) * TT);
	double *ftmp = malloc(sizeof(double) * TT);
	double *c = malloc(sizeof(double) * TT);
	double *tmp5 = malloc(sizeof(double) * TT*p);
	double *hpsi = malloc(sizeof(double) * TT);
	double *psi = malloc(sizeof(double) * TT);
	double *gam2 = malloc(sizeof(double) * TT);
	double *gam_tmp = malloc(sizeof(double) * TT);
	double *max_val = malloc(sizeof(double) * (max_itr+1));
	double *tmp7 = malloc(sizeof(double) * TT);
	double O1[4];
	double O_tmp[4], O2[4];
	double binsize, A, theta, B, tmp1, tmp2, thetanew, tmpi;
	double max_val_change, res_cos, res_sin, hO;

	// Pointers
	double *t_ptr, *gam1_ptr, *f_basis_ptr, *q_tilde_ptr, *A_ptr, *nu_ptr;
	double *O_tmp_ptr, *q_tmp_ptr, *alpha_ptr, *q_tilde_diff_ptr;
	double *ftmp_ptr, *c_ptr, *cbar_ptr, *tmp5_ptr, *hpsi_ptr, *psi_ptr;
	double *ones_ptr, *gam2_ptr, *gam_tmp_ptr, *tmp7_ptr, *O1_ptr, *q_ptr;
	int *y_ptr;

	t_ptr = t;
	linspace(0, 1, TT, t_ptr);
	binsize = 1.0/(TT-1);

	gam1_ptr = gam1;
	for (k=0; k<TT; k++){
		gam1[k] = t[k];
		ones[k] = 1.0;
	}

	O1[0] = 1;
	O1[1] = 0;
	O1[2] = 0;
	O1[3] = 1;

	// warping basis (fourier)
	f_basis_ptr = f_basis;
	for (k=0; k<p1; k++){
		for (j=0; j<TT; j++){
			f_basis[j+2*k*TT] = 1/sqrt(M_PI) * sin(2*M_PI*(k+1)*t[j]);
			f_basis[j+(2*k+1)*TT] = 1/sqrt(M_PI) * cos(2*M_PI*(k+1)*t[j]);
		}
	}

	q_tilde_ptr = q_tilde;
	for (k=0; k<TT*n; k++)
		q_tilde[k] = q[k];

	do {
		// inner product value
		nu_ptr = nu;
		q_tilde_ptr = q_tilde;
		A = innerprod_q2(T1, q_tilde_ptr, nu_ptr);

		// for gradient for rotation
		theta = acos(O1[0]);
		O_tmp[0] = -1*sin(theta);
		O_tmp[1] = cos(theta);
		O_tmp[2] = -1*cos(theta);
		O_tmp[3] = -1*sin(theta);

		O_tmp_ptr = O_tmp;
		q_tilde_ptr = q_tilde;
		q_tmp_ptr = q_tmp;
		product(n, n, TT, O_tmp_ptr, q_tilde_ptr, q_tmp_ptr);
		nu_ptr = nu;
		B = innerprod_q2(T1, q_tmp_ptr, nu_ptr);

		y_ptr = y;
		alpha_ptr = alpha;
		tmp1 = exp((-1*y_ptr[0])*(alpha_ptr[0] + A));
		tmp2 = (y_ptr[0]*tmp1)/(1+tmp1);

		hO = tmp2*B;

		thetanew = theta+deltaO*hO;
		O2[0] = cos(thetanew);
		O2[1] = sin(thetanew);
		O2[2] = -1*sin(thetanew);
		O2[3] = cos(thetanew);

		// form graident for warping
		q_tilde_diff_ptr = q_tilde_diff;
		q_tilde_ptr = q_tilde;
		col_gradient(n, TT, q_tilde_ptr, binsize, q_tilde_diff_ptr);

		cbar_ptr = cbar;
		ftmp_ptr = ftmp;
		q_tmp_ptr = q_tmp;
		c_ptr = c;
		nu_ptr = nu;
		tmp5_ptr = tmp5;
		for (k=0; k<p; k++){
			for (l=0; l<TT; l++)
				ftmp[l] = f_basis[l+k*TT];
			cumtrapz(T1, t_ptr, ftmp_ptr, cbar_ptr);
			for (jj=0; jj<n; jj++){
				for (l=0; l<TT; l++)
					q_tmp[n*l+jj] = 2*q_tilde_diff[n*l+jj]*cbar[l] + q_tilde[n*l+jj]*f_basis[l+k*TT];
			}

			tmpi = innerprod_q2(T1, q_tmp_ptr, nu_ptr);
			for (l=0; l<TT; l++)
				tmp5[l+k*TT] = tmpi*f_basis[l+k*TT];
		}

		for (jj=0; jj<TT; jj++){
			c[jj] = tmp5[jj];
			for (l=1; l<p; l++)
				c[jj] = c[jj] + tmp5[jj+l*TT];
		}

		hpsi_ptr = hpsi;
		c_ptr = c;
		for (k=0; k<TT; k++)
			hpsi_ptr[k] = tmp2 * c_ptr[k];

		psi_ptr = psi;
		ones_ptr = ones;
		gam2_ptr = gam2;
		gam_tmp_ptr = gam_tmp;
		hpsi_ptr = hpsi;
		pvecnorm(T1, hpsi_ptr, &binsize1, &tmpi);

		res_cos = cos(deltag*tmpi);
		res_sin = sin(deltag*tmpi);
		if (tmpi == 0){
			for (j=0; j<TT; j++)
				psi_ptr[j] = res_cos*ones_ptr[j];
		}
		else{
			for (j=0; j<TT; j++)
				psi_ptr[j] = res_cos*ones_ptr[j] + res_sin*(hpsi_ptr[j]/tmpi);
		}

		for (j=0; j<TT; j++)
			tmp7[j] = psi_ptr[j]*psi_ptr[j];

		tmp7_ptr = tmp7;
		cumtrapz(T1,t_ptr,tmp7_ptr,gam_tmp_ptr);
		for (j=0; j<TT; j++)
			gam_tmp_ptr[j] = (gam_tmp_ptr[j] - gam_tmp_ptr[0])/(gam_tmp_ptr[TT-1]-gam_tmp_ptr[0]); // slight change of scale

		approx(t, gam1_ptr, TT, gam_tmp_ptr, gam2_ptr, TT, 1, 0, 1, 0);

		y_ptr = y;
		alpha_ptr = alpha;
		max_val[itr] = log(1/(1+exp((-1*y_ptr[0])*(alpha_ptr[0]+A))));

		if (display == 1)
			printf("Iteration %d : Cost %f\n", (itr+1), max_val[itr]);

		gam1_ptr = gam2;
		O1_ptr = O2;
		O1[0] = O2[0];
		O1[1] = O2[1];
		O1[2] = O2[2];
		O1[3] = O2[3];

		q_tilde_ptr = q_tilde;
		q_tmp_ptr = q_tmp;
		q_ptr = q;
		product(n, n, TT, O1_ptr, q_ptr, q_tmp_ptr);
		group_action_by_gamma(n1, T1, q_tmp_ptr, gam1_ptr, q_tilde_ptr);

		if (itr >= 2){
			max_val_change = max_val[itr] - max_val[itr-1];
			if (hO < tol && tmpi < tol)
				break;
		}

		itr++;

	} while (max_itr>=itr);

	for (k=0; k<TT; k++){
		gamout[k] = gam1_ptr[k];
	}

	for (k=0; k<n*n; k++){
		Oout[k] = O1_ptr[k];
	}

	free(t);
	free(gam1);
	free(ones);
	free(f_basis);
	free(q_tilde);
	free(q_tmp);
	free(q_tilde_diff);
	free(cbar);
	free(ftmp);
	free(c);
	free(tmp5);
	free(hpsi);
	free(psi);
	free(gam2);
	free(gam_tmp);
	free(max_val);
	free(tmp7);

}
