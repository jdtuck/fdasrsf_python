#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include "misc_funcs.h"
#include "matrix_exponential.h"

void ocmlogit_warp_grad(int *n1, int *T1, int *m1, double *alpha, double *nu, double *q, int *y, int *max_itri, double *toli, double *deltaOi, double *deltagi, int *displayi, double *gamout, double *Oout){

	/* dereference inputs
	alpha, beta, q should be normalized by norm */
	int TT = *T1;  // number of sample points
	int m = *m1;  // number of classes
	int n = *n1;  // number of dimension of curves R^n
	int max_itr = *max_itri;
	double tol = *toli, deltaO = *deltaOi, deltag = *deltagi;
	double display = *displayi;

	// Looping and temp variables
	int k, j, l, kk, jj;
	int p = 20;
	int p1 = 10;
	int itr = 1;
	double binsize1 = 1.0;
	double t[TT], O1[4], O2[4], binsize, E[4], A[m], O_tmp[4];
	double gam1[TT], f_basis[TT*p], max_val[max_itr];
	double q_tilde[TT*n], B[m], q_tmp[TT*n];
	double tmpi, tmp1, tmp2[m], tmp3, tmp4, tmpi1;
	double hO, q_tilde_diff[TT*n], c[TT*m], cbar[TT], ftmp[TT];
	double tmp5[TT*p], tmp6[TT*m], tmp7[TT], tmp8[TT];
	double hpsi[TT], ones[TT], psi[TT], gam2[TT], gam_tmp[TT];
	double max_val_change, res_cos, res_sin, theta;

	// Pointers
	double *t_ptr, *f_basis_ptr, *A_ptr, *tmpi_ptr, *q_tilde_ptr;
	double *nu_ptr, *B_ptr, *E_ptr, *q_tmp_ptr, *alpha_ptr;
	double *tmp2_ptr, *tmp3_ptr, *O1_ptr, *O2_ptr, *tmp4_ptr;
	double *hO_ptr, *O_tmp_ptr, *q_tilde_diff_ptr, *c_ptr, *cbar_ptr;
	double *ftmp_ptr, *tmp5_ptr, *tmp6_ptr, *tmp7_ptr, *tmp8_ptr;
	double *hpsi_ptr, *psi_ptr, *gam1_ptr, *gam2_ptr, *ones_ptr;
	double *gam_tmp_ptr, *q_ptr;
	int *y_ptr;

	t_ptr = t;
	linspace(0, 1, TT, t_ptr);
	binsize = 1/(TT-1);

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
		A_ptr = A;
		nu_ptr = nu;
		q_tilde_ptr = q_tilde;
		for (j=0; j<m; j++){
			A[j] = innerprod_q2(T1, q_tilde_ptr, nu_ptr);
			nu_ptr += n*TT;
			q_tilde_ptr += n*TT;
		}

		// for gradient for rotation
		theta = acos(O1[0]);
		O_tmp[0] = -1*sin(theta);
		O_tmp[1] = -1*cos(theta);
		O_tmp[2] = cos(theta);
		O_tmp[3] = -1*sin(theta);

		B_ptr = B;
		O_tmp_ptr = O_tmp;
		q_tilde_ptr = q_tilde;
		q_tmp_ptr = q_tmp;
		product(n, n, TT, O_tmp_ptr, q_tilde_ptr, q_tmp_ptr);
		for (j=0; j<m; j++){
			B[j] = innerprod_q2(T1, q_tmp_ptr, nu_ptr);
			nu_ptr += n*TT;
		}

		tmp1 = 0;
		A_ptr = A;
		alpha_ptr = alpha;
		for (j=0; j<m; j++)
			tmp1 += exp(alpha_ptr[j] + A_ptr[j]);

		tmp2_ptr = tmp2;
		B_ptr = B;
		for (j=0; j<m; j++){
			tmp2_ptr[k] = exp(alpha_ptr[j] + A_ptr[j]) * B_ptr[k];
		}

		tmp2_ptr = tmp2;
		tmp3 = 0;
		for (k=0; k<m; k++){
			tmp3 += tmp2_ptr[k];
		}

		tmp3 = tmp3/tmp1;

		tmp4 = 0;
		B_ptr = B;
		y_ptr = y;
		for (k=0; k<m; j++){
			tmp4 += y_ptr[k] * B_ptr[k];
		}

		hO = tmp4 - tmp3;

		O2[0] = cos(theta+deltaO*hO);
		O2[1] = -1*sin(theta+deltaO*hO);
		O2[2] = sin(theta+deltaO*hO);
		O2[3] = cos(theta+deltaO*hO);

		// form graident for warping
		q_tilde_diff_ptr = q_tilde_diff;
		q_tilde_ptr = q_tilde;
		col_gradient(n, TT, q_tilde_ptr, binsize, q_tilde_diff_ptr);

		cbar_ptr = cbar;
		ftmp_ptr = ftmp;
		q_tmp_ptr = q_tmp;
		c_ptr = c;
		nu_ptr = nu;
		for (j=0; j<m; j++){
			tmp5_ptr = tmp5;
			for (k=0; k<p; k++){
				for (l=0; l<TT; l++)
					ftmp[l] = f_basis[l+k*TT];
				cumtrapz(T1, t_ptr, ftmp_ptr, cbar_ptr);
				for (jj=0; jj<n; jj++){
					for (kk=0; kk<TT; kk++){
						for (l=0; l<TT; l++)
							q_tmp[n*l+jj] = 2*q_tilde_diff[n*l+jj]*cbar[kk] + q_tilde[n*l+jj]*f_basis[l+k*TT];
					}
				}

				tmpi = innerprod_q2(T1, q_tmp_ptr, nu_ptr);
				for (l=0; l<TT; l++)
					tmp5[l+k*TT] = tmpi*f_basis[l+k*TT];
			}


			for (jj=0; jj<TT; jj++){
				c[jj+TT*j] = tmp5[jj];
				for (l=1; l<p; l++)
					c[jj+TT*j] = c[jj+TT*j] + tmp5[l*TT];
			}

			nu_ptr += n*TT;
		}

		tmp6_ptr = tmp6;
		c_ptr = c;
		A_ptr = A;
		for (j=0; j<m; j++){
			for (k=0; k<TT; k++){
				tmp6_ptr[k] = exp(alpha_ptr[j] + A_ptr[j]) * c_ptr[k];
			}
			c_ptr += TT;
			tmp6_ptr += TT;
		}

		tmp7_ptr = tmp7;
		tmp6_ptr = tmp6;
		for (k=0; k<TT; k++){
			tmp7[k] = tmp6_ptr[k];
			for (j=1; j<m; j++)
				tmp7[k] = tmp7[k] + tmp6_ptr[k+j*n*n];
		}

		for (k=0; k<n*n; k++)
			tmp7[k] = tmp7[k]/tmp1;

		tmp8_ptr = tmp8;
		c_ptr = c;
		y_ptr = y;
		for (j=0; j<m; j++){
			for (k=0; k<TT; k++){
				tmp8_ptr[k] = y_ptr[j] * c_ptr[k];
			}
			c_ptr += TT;
			tmp8_ptr += TT;
		}

		hpsi_ptr = hpsi;
		tmp8_ptr = tmp8;
		for (k=0; k<TT; k++){
			hpsi[k] = tmp8_ptr[k];
			for (j=1; j<m; j++)
				hpsi[k] += tmp8_ptr[k+j*n*n];
		}

		tmp7_ptr = tmp7;
		for (k=0; k<TT; k++)
			hpsi[k] = hpsi[k] - tmp7_ptr[k];

		psi_ptr = psi;
		ones_ptr = ones;
		gam2_ptr = gam2;
		gam_tmp_ptr = gam_tmp;
		pvecnorm(T1, hpsi, &binsize1, &tmpi);
		res_cos = cos(deltag*tmpi);
		res_sin = sin(deltag*tmpi);
		for (j=0; j<TT; j++)
			psi_ptr[j] = res_cos*ones_ptr[j] + res_sin*(hpsi_ptr[j]/tmpi);

		for (j=0; j<TT; j++)
			tmp8[j] = psi_ptr[j]*psi_ptr[j];

		tmp8_ptr = tmp8;
		cumtrapz(T1,t_ptr,tmp8_ptr,gam_tmp_ptr);
		for (j=0; j<TT; j++)
			gam_tmp_ptr[j] = (gam_tmp_ptr[j] - gam_tmp_ptr[0])/(gam_tmp_ptr[TT-1]-gam_tmp_ptr[0]); // slight change of scale

		approx(t, gam1_ptr, TT, gam_tmp_ptr, gam2_ptr, TT, 1, 0, 1, 0);

		tmpi1 = 0;
		alpha_ptr = alpha;
		A_ptr = A;
		for (j=0; j<m; j++)
			tmpi1 += y[j] * (alpha_ptr[j] + A_ptr[j]);

		max_val[itr] = tmpi1 - log(tmp1);

		if (display == 1)
			printf("Iteration %d : Cost %f\n", itr, max_val[itr]);

		gam1_ptr = gam2;
		O1_ptr = O2;

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

	for (k=1; k<TT; k++){
		gamout[k] = gam1_ptr[k];
	}

	for (k=1; k<n*n; k++){
		Oout[k] = O1_ptr[k];
	}

}
