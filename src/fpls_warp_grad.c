#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include "misc_funcs.h"

void fpls_warp_grad(int *m1, int *n1, double *ti, double *gami, double *qf, double *qg, double *wf, double *wg, 
	int *max_itri, double *toli, double *deltai, int *displayi, double *gamout){
	
	
	// dereference inputs
	const int TT = *m1;
	int N = *n1, max_itr = *max_itri;
	double t = *ti, gam = *gami;
	double tol = *toli, delta = *deltai, display = *displayi;

	// Looping and temp variables
	int k, j;
	int n2 = 1;
	int itr = 1;
	double tmp2 = 0;
	double N1 = N;
	const int size_array = TT*N;
	double *psi1 = malloc(sizeof(double)*size_array);
	double *gam2 = malloc(sizeof(double)*size_array);
	double *rfi_diff = malloc(sizeof(double)*TT);
	double *rgi_diff = malloc(sizeof(double)*TT);
	double *grad = malloc(sizeof(double)*TT);
	double *vec = malloc(sizeof(double)*TT);
	double *gamI = malloc(sizeof(double)*TT);
	double eps = DBL_EPSILON;
	double *tmp = malloc(sizeof(double)*TT);
	double *psi2 = malloc(sizeof(double)*size_array);
	double *xout = malloc(sizeof(double)*TT);
	double *qf_tmp = malloc(sizeof(double)*size_array);
	double *qg_tmp = malloc(sizeof(double)*size_array);
	double binsize;
	double *rfi = malloc(sizeof(double)*N);
	double *rgi = malloc(sizeof(double)*N);
	double *qf_tmp_diff = malloc(sizeof(double)*size_array);
	double *qg_tmp_diff = malloc(sizeof(double)*size_array);
	double *gam1 = malloc(sizeof(double)*size_array);
	double *max_val = malloc(sizeof(double)*max_itr);
	double tmpi, tmpj;
	double res_cos, res_sin, max_val_change;
	double *tmp1 = malloc(sizeof(double)*(TT));

	// Pointers
	double *qf_ptr, *qg_ptr, *gam_ptr, *psi_ptr, *gam2_ptr, *psi2_ptr;
	double *qf_tmp_ptr, *qg_tmp_ptr, *qf_tmp_diff_ptr, *qg_tmp_diff_ptr;
	double *xout_ptr, *tmp_ptr, *tmp1_ptr, *tmpi_ptr, *tmpj_ptr;
	double *grad_ptr, *rfi_ptr, *rgi_ptr, *gamI_ptr;
	
	binsize = 0;
	for (k=0; k<TT-1; k++)
		binsize += ti[k+1]-ti[k];
	binsize = binsize/(TT-1);

	for (k=0; k<TT*N; k++){
		gam1[k] = gami[k];
	}
	psi_ptr = psi1; gam_ptr = gam1;
	gradient(m1,n1,gam_ptr,&binsize,psi_ptr);
	for (k=0; k<TT*N; k++)
		psi1[k] = sqrt(fabs(psi1[k])+eps);

	do {
		qf_ptr = qf; qf_tmp_ptr = qf_tmp; 
		qg_ptr = qg; qg_tmp_ptr = qg_tmp;
		qf_tmp_diff_ptr = qf_tmp_diff; qg_tmp_diff_ptr = qg_tmp_diff;
		for (k=0; k<N; k++) {
			for (j=0; j<TT; j++)
				xout[j] = (ti[TT-1] - ti[0])*gam_ptr[j]+ti[0];
			xout_ptr = xout;
			spline(TT, ti, qf_ptr, TT, xout_ptr, qf_tmp_ptr);
			spline(TT, ti, qg_ptr, TT, xout_ptr, qg_tmp_ptr);

			for (j=0; j<TT; j++)
				tmp[j] = qf_tmp_ptr[j]*psi_ptr[j];
			tmp_ptr = tmp;
			innerprod_q(m1, ti, tmp_ptr, wf, &tmpi); rfi[k] = tmpi;
			for (j=0; j<TT; j++)
				tmp[j] = qg_tmp_ptr[j]*psi_ptr[j];
			tmp_ptr = tmp;
			innerprod_q(m1, ti, tmp_ptr, wg, &tmpi); rgi[k] = tmpi;

			tmp_ptr = tmp;
			gradient(m1,&n2,qf_ptr,&binsize,tmp_ptr);
			spline(TT, ti, tmp_ptr, TT, xout_ptr, qf_tmp_diff_ptr);
			tmp_ptr = tmp;
			gradient(m1,&n2,qg_ptr,&binsize,tmp_ptr);
			spline(TT, ti, tmp_ptr, TT, xout_ptr, qg_tmp_diff_ptr);

			qf_ptr += TT;
			qf_tmp_ptr += TT;
			qg_ptr += TT;
			qg_tmp_ptr += TT;
			gam_ptr += TT;
			psi_ptr += TT;
			qf_tmp_diff_ptr += TT;
			qg_tmp_diff_ptr += TT;
		}

		psi_ptr = psi_ptr - TT*N;
		qf_tmp_diff_ptr = qf_tmp_diff; qg_tmp_diff_ptr = qg_tmp_diff;
		qg_tmp_ptr = qg_tmp; qf_tmp_ptr = qf_tmp; 
		psi2_ptr = psi2; gam2_ptr = gam2;
		for (k=0; k<N; k++) {
			tmp_ptr = tmp;
			for (j=0; j<TT; j++)
				tmp_ptr[j] = qf_tmp_diff_ptr[j]*psi_ptr[j]*wf[j];
			tmp1_ptr = tmp1;
			tmpi_ptr = &tmpi;
			trapz(&TT, &n2, ti, tmp_ptr, tmpi_ptr);
			cumtrapz(&TT, ti, tmp_ptr, tmp1_ptr);
			for (j=0; j<TT; j++)
				tmp1[j] = tmpi - tmp1[j];
			for (j=0; j<TT; j++)
				rfi_diff[j] = 2*psi_ptr[j]*tmp1[j]+qf_tmp_ptr[j]*wf[j];

			tmp_ptr = tmp;
			for (j=0; j<TT; j++)
				tmp_ptr[j] = qg_tmp_diff_ptr[j]*psi_ptr[j]*wg[j];
			tmp1_ptr = tmp1;
			tmpi_ptr = &tmpi;
			trapz(&TT, &n2, ti, tmp_ptr, tmpi_ptr);
			cumtrapz(&TT, ti, tmp_ptr, tmp1_ptr);
			for (j=0; j<TT; j++)
				tmp1[j] = tmpi - tmp1[j];
			for (j=0; j<TT; j++)
				rgi_diff[j] = 2*psi_ptr[j]*tmp1[j]+qg_tmp_ptr[j]*wg[j];

			tmpi = 0;
			tmpj = 0;
			for (j=0; j<N; j++){
				if (j == k)
					continue;

				tmpi += rfi[j];
				tmpj += rgi[j];
			}
			for (j=0; j<TT; j++)
				grad[j] = 1/N1*rfi_diff[j]*rgi[k]+1/N1*rfi[k]*rgi_diff[j] - 1/(N1*N1)*rfi_diff[j]*rgi[k]-1/(N1*N1)*rfi[k]*rgi_diff[j] - 1/(N1*N1)*rfi_diff[j]*tmpj- 1/(N1*N1)*rgi_diff[j]*tmpi;

			grad_ptr = grad;
			tmpi_ptr = &tmpi;
			innerprod_q(m1, ti, grad_ptr, psi_ptr, tmpi_ptr);
			for (j=0; j<TT; j++)
				vec[j] = grad[j] - tmpi*psi_ptr[j];

			pvecnorm2(&TT, vec, &binsize, &tmpi);
			res_cos = cos(delta*tmpi);
			res_sin = sin(delta*tmpi);
			for (j=0; j<TT; j++)
				psi2_ptr[j] = res_cos*psi_ptr[j] + res_sin*(vec[j]/tmpi);
			
			innerprod_q(m1, ti, psi2_ptr, psi2_ptr, &tmpi);
			for (j=0; j<TT; j++)
				psi2_ptr[j] = psi2_ptr[j]/tmpi;
				
			for (j=0; j<TT; j++)
				tmp1[j] = psi2_ptr[j]*psi2_ptr[j];
			
			tmp1_ptr = tmp1;
			cumtrapz(&TT,ti,tmp1_ptr,gam2_ptr);
			for (j=0; j<TT; j++)
				gam2_ptr[j] = (gam2_ptr[j] - gam2_ptr[0])/(gam2_ptr[TT-1]-gam2_ptr[0]); // slight change of scale

			qf_tmp_diff_ptr += TT;
			qg_tmp_diff_ptr += TT;
			psi_ptr += TT;
			qf_tmp_ptr += TT;
			qg_tmp_ptr += TT;
			psi2_ptr += TT;
			gam2_ptr += TT;

		}

		rfi_ptr = rfi;
		rgi_ptr = rgi;
		tmpi_ptr = &tmpi;
		cov(N, rfi_ptr, rgi_ptr, tmpi_ptr);
		max_val[itr] = tmpi;

		if (display == 1)
			printf("Iteration %d : Cost %f\n", itr, max_val[itr]);

		gam_ptr = gam2;
		// gamI_ptr = gamI;
		// SqrtMeanInverse(m1, n1, ti, gam_ptr, gamI_ptr);

		// gam2_ptr = gam1;
		// for (k=0; k<N; k++) {
		// 	for (j=0; j<TT; j++)
		// 		xout[j] = (ti[TT-1] - ti[0])*gamI_ptr[j]+ti[0];
		// 	xout_ptr = xout;
		// 	approx(ti, gam_ptr, TT, xout_ptr, gam2_ptr, TT, 1, 0, 1, 0);

		// 	gam_ptr += TT;
		// 	gam2_ptr += TT;
		// }

		// gam_ptr = gam1;
		// psi_ptr = psi2;
		// gradient(m1,n1,gam_ptr,&binsize,psi_ptr);
		// for (k=0; k<TT*N; k++)
		// 	psi2[k] = sqrt(fabs(psi2[k])+eps);
		psi_ptr = psi2;

		if (itr >= 2){
			max_val_change = max_val[itr] - max_val[itr-1];
			if (fabs(max_val_change) < tol)
				break;
			if (max_val_change < 0)
				break;
		}

		itr++;

	} while (max_itr>itr);

	for (k=0; k<TT*N; k++){
		gamout[k] = gam_ptr[k];
	}

	free(psi1);
	free(gam2);
	free(rfi_diff);
	free(rgi_diff);
	free(grad);
	free(vec);
	free(gamI);
	free(tmp);
	free(psi2);
	free(xout);
	free(qf_tmp);
	free(qg_tmp);
	free(rfi);
	free(rgi);
	free(qf_tmp_diff);
	free(qg_tmp_diff);
	free(gam1);
	free(max_val);
	free(tmp1);
}