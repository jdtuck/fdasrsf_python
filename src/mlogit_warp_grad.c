#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include "misc_funcs.h"

void mlogit_warp_grad(int *m1, int *m2, double *alpha, double *beta, double *ti, double *gami, double *q, int *y, int *max_itri, double *toli, double *deltai, int *displayi, double *gamout)
{

	// dereference inputs
	// alpha, beta, q should be normalized by norm
	int TT = *m1;
	int m = *m2;
	int max_itr = *max_itri;
	double t = *ti;
	double tol = *toli, delta = *deltai, display = *displayi;

	// Looping and temp variables
	int k, j;
	int n1 = 1;
	int itr = 1;
	double *gam1 = malloc(sizeof(double) * TT);
	double *psi1 = malloc(sizeof(double) * TT);
	double *q_tmp = malloc(sizeof(double) * TT);
	double *q_tmp_diff = malloc(sizeof(double) * TT);
	double *A = malloc(sizeof(double) * m);
	double *Adiff = malloc(sizeof(double) * TT*m);
	double *xout = malloc(sizeof(double) * TT);
	double *tmp = malloc(sizeof(double) * TT);
	double *tmp3 = malloc(sizeof(double) * TT*m);
	double *h = malloc(sizeof(double) * TT);
	double *vec = malloc(sizeof(double) * TT);
	double *psi2 = malloc(sizeof(double) * TT);
	double *gam2 = malloc(sizeof(double) * TT);
	double *max_val = malloc(sizeof(double) * max_itr);
	double tmp1, tmpi, binsize;
	double eps = DBL_EPSILON;
	double res_cos, res_sin, max_val_change;
	double *tmp2 = malloc(sizeof(double) * (TT));

	// Pointers
	double *psi_ptr, *gam_ptr, *q_ptr, *q_tmp_ptr, *q_tmp_diff_ptr;
	double *xout_ptr, *tmp_ptr, *A_ptr, *Adiff_ptr, *tmp1_ptr, *tmp2_ptr;
	double *alpha_ptr, *beta_ptr, *tmp3_ptr, *h_ptr;
	int *y_ptr;
	double *psi2_ptr, *gam2_ptr, *tmpi_ptr;

	binsize = 0;
	for (k = 0; k < TT - 1; k++)
		binsize += ti[k + 1] - ti[k];
	binsize = binsize / (TT - 1);

	for (k = 0; k < TT; k++)
	{
		gam1[k] = gami[k];
	}
	psi_ptr = psi1;
	gam_ptr = gam1;
	gradient(m1, &n1, gam_ptr, &binsize, psi_ptr);
	for (k = 0; k < TT; k++)
		psi1[k] = sqrt(fabs(psi1[k]) + eps);

	do
	{
		q_ptr = q;
		q_tmp_ptr = q_tmp;
		q_tmp_diff_ptr = q_tmp_diff;
		alpha_ptr = alpha;
		beta_ptr = beta;

		for (j = 0; j < TT; j++)
			xout[j] = (ti[TT - 1] - ti[0]) * gam_ptr[j] + ti[0];
		xout_ptr = xout;
		spline(TT, ti, q_ptr, TT, xout_ptr, q_tmp_ptr);

		tmp_ptr = tmp;
		gradient(m1, &n1, q_ptr, &binsize, tmp_ptr);
		spline(TT, ti, tmp_ptr, TT, xout_ptr, q_tmp_diff_ptr);

		A_ptr = A;
		Adiff_ptr = Adiff;
		tmp2_ptr = tmp2;
		tmp1_ptr = &tmp1;
		for (j = 0; j < m; j++)
		{
			for (k = 0; k < TT; k++)
				tmp[k] = q_tmp_ptr[k] * psi_ptr[k] * beta_ptr[k];
			tmp_ptr = tmp;
			trapz(m1, &n1, ti, tmp_ptr, tmp1_ptr);
			A[j] = tmp1;
			for (k = 0; k < TT; k++)
				tmp[k] = q_tmp_diff_ptr[k] * psi_ptr[k] * beta_ptr[k];
			tmp_ptr = tmp;
			trapz(&TT, &n1, ti, tmp_ptr, tmp1_ptr);
			cumtrapz(&TT, ti, tmp_ptr, tmp2_ptr);
			for (k = 0; k < TT; k++)
				tmp[k] = tmp1_ptr[0] - tmp2_ptr[k];
			tmp_ptr = tmp;
			for (k = 0; k < TT; k++)
				Adiff_ptr[k] = 2 * psi_ptr[k] * tmp_ptr[k] + q_tmp_ptr[k] * beta_ptr[k];

			beta_ptr += TT;
			Adiff_ptr += TT;
		}

		tmp1 = 0;
		for (j = 0; j < m; j++)
			tmp1 += exp(alpha_ptr[j] + A_ptr[j]);

		tmp3_ptr = tmp3;
		Adiff_ptr = Adiff;
		for (j = 0; j < m; j++)
		{
			for (k = 0; k < TT; k++)
			{
				tmp3_ptr[k] = exp(alpha_ptr[j] + A_ptr[j]) * Adiff_ptr[k];
			}
			Adiff_ptr += TT;
			tmp3_ptr += TT;
		}

		tmp_ptr = tmp;
		tmp3_ptr = tmp3;
		for (k = 0; k < TT; k++)
		{
			tmp[k] = tmp3_ptr[k];
			for (j = 1; j < m; j++)
				tmp[k] = tmp[k] + tmp3_ptr[k + j * TT];
		}

		for (k = 0; k < TT; k++)
			tmp[k] = tmp[k] / tmp1;

		tmp3_ptr = tmp3;
		Adiff_ptr = Adiff;
		y_ptr = y;
		for (j = 0; j < m; j++)
		{
			for (k = 0; k < TT; k++)
			{
				tmp3_ptr[k] = y_ptr[j] * Adiff_ptr[k];
			}
			Adiff_ptr += TT;
			tmp3_ptr += TT;
		}

		h_ptr = h;
		tmp3_ptr = tmp3;
		for (k = 0; k < TT; k++)
		{
			h[k] = tmp3_ptr[k];
			for (j = 1; j < m; j++)
				h[k] += tmp3_ptr[k + j * TT];
		}

		tmp_ptr = tmp;
		for (k = 0; k < TT; k++)
			h[k] = h[k] - tmp_ptr[k];

		tmpi_ptr = &tmpi;
		innerprod_q(m1, ti, h_ptr, psi_ptr, tmpi_ptr);

		for (j = 0; j < TT; j++)
			vec[j] = h[j] - tmpi * psi_ptr[j];

		psi2_ptr = psi2;
		gam2_ptr = gam2;
		pvecnorm(&TT, vec, &binsize, &tmpi);
		res_cos = cos(delta * tmpi);
		res_sin = sin(delta * tmpi);
		for (j = 0; j < TT; j++)
			psi2_ptr[j] = res_cos * psi_ptr[j] + res_sin * (vec[j] / tmpi);

		for (j = 0; j < TT; j++)
			tmp[j] = psi2_ptr[j] * psi2_ptr[j];

		tmp_ptr = tmp;
		cumtrapz(&TT, ti, tmp_ptr, gam2_ptr);
		for (j = 0; j < TT; j++)
			gam2_ptr[j] = (gam2_ptr[j] - gam2_ptr[0]) / (gam2_ptr[TT - 1] - gam2_ptr[0]); // slight change of scale

		tmpi = 0;
		alpha_ptr = alpha;
		A_ptr = A;
		for (j = 0; j < m; j++)
			tmpi += y[j] * (alpha_ptr[j] + A_ptr[j]);

		max_val[itr] = tmpi - log(tmp1);

		if (display == 1)
			printf("Iteration %d : Cost %f\n", itr, max_val[itr]);

		gam_ptr = gam2;
		psi_ptr = psi2;

		if (itr >= 2)
		{
			max_val_change = max_val[itr] - max_val[itr - 1];
			if (fabs(max_val_change) < tol)
				break;
			if (max_val_change < 0)
				break;
		}

		itr++;

	} while (max_itr >= itr);

	for (k = 1; k < TT; k++)
	{
		gamout[k] = gam2_ptr[k];
	}

	free(gam1);
	free(psi1);
	free(q_tmp);
	free(q_tmp_diff);
	free(A);
	free(Adiff);
	free(xout);
	free(tmp);
	free(tmp3);
	free(h);
	free(vec);
	free(psi2);
	free(gam2);
	free(max_val);
	free(tmp2);
}
