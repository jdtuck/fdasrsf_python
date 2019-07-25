/*
This file defines the class for the problem
min_X tr(X^T B X D), where B is a symmetric matrix, D is a diagonal matrix and X \in St(p, n).

Problem --> StieBrockett

---- WH
*/

#ifndef STIEBROCKETT_H
#define STIEBROCKETT_H

#include "Stiefel.h"
#include "StieVariable.h"
#include "StieVector.h"
#include "Problem.h"
#include "SharedSpace.h"
#include "def.h"
#include "MyMatrix.h"

/*Define the namespace*/
namespace ROPTLIB{

	class StieBrockett : public Problem{
	public:
		StieBrockett(double *inB, double *inD, integer inn, integer inp);
		virtual ~StieBrockett();
		virtual double f(Variable *x) const;

		//virtual void RieGrad(Variable *x, Vector *gf) const;
		//virtual void RieHessianEta(Variable *x, Vector *etax, Vector *xix) const;

		virtual void EucGrad(Variable *x, Vector *egf) const;
		virtual void EucHessianEta(Variable *x, Vector *etax, Vector *exix) const;

		double *B;
		double *D;
		integer n;
		integer p;
	};
}; /*end of ROPTLIB namespace*/
#endif // end of STIEBROCKETT_H
