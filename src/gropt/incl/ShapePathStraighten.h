/*
 This file defines the class for the problem in [YHGA2015]
 [YHGA2015]: Yaqing You and Wen Huang and Kyle A. Gallivan and P.-A. Absil, A Riemannian Approach for Computing Geodesics in Elastic Shape Analysis, in Proceedings of the 3rd IEEE Global Conference on Signal & Information Processing, 2015
 Find a geodesic between two closed curves in shape space.

 Problem -->ShapePathStraighten

 ----- YY
 */

#ifndef ShapePathStraighten_h
#define ShapePathStraighten_h

#include "PreShapeCurves.h"
#include "PreShapePathStraighten.h"
#include "SCVariable.h"
#include "PSCVector.h"
#include "ElasticCurvesRO.h"
#include "RSD.h"
#include "olversLS.h"

#include "def.h"
#include "Problem.h"
#include "SharedSpace.h"
#include "Spline.h"
#include "MyMatrix.h"

/*Define the namespace*/
namespace ROPTLIB{
	class ShapePathStraighten : public Problem {
	public:
		ShapePathStraighten(double *inq1, double *inq2, integer innumP, integer indim, integer innumC);

		/*Destructor*/
		virtual ~ShapePathStraighten();

		/*cost function*/
		virtual double f(Variable *x) const;

		/*Euclidean gradient*/
		virtual void EucGrad(Variable *x, Vector *egf) const;

		/*Apply given rotation, reparameterization and m to any closed curve q2*/
		static void Apply_Oml(const double *O, const double *m, const double *l, integer innumP, integer indim, const double *q_2_spline_coeff, double *q2_new);

        mutable PSCVariable *finalPSCV;
        //        //===================== For Splines =====================
        mutable double *log_map;
        //        //===================== For Splines =====================


	private:
		integer numP; //Number of points on each curve
		integer dim; //Dimension of the space where the curve is in
		integer numC; //Number of curves on each path

		double *q1;			// n by d matrices
		double *q2_coefs;	// (n - 1) by 4 d matrices, d cubic splines
		double *dq2_coefs;	// (n - 1) by 3 d matrices, d cubic splines
	};
}; /*end of ROPTLIB namespace*/

#endif /* ShapePathStraighten_h */
