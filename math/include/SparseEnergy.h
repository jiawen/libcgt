#ifndef SPARSE_ENERGY_H
#define SPARSE_ENERGY_H

#include <cholmod.h>

class FloatMatrix;

// interface
class SparseEnergy
{
public:

	virtual int numFunctions() = 0;
	virtual int numVariables() = 0;
	virtual int maxNumNonZeroes() = 0;

	virtual void evaluateInitialGuess( FloatMatrix& guess ) = 0;

	// Evaluate the energy at argument beta (a numVariables x 1 vector)
	// and return it in r (a numFunctions x 1 vector)
	virtual void evaluateResidual( const FloatMatrix& beta, FloatMatrix& residual ) = 0;

	// Evaluate the Jacobian \J_r at argument \Beta
	// J(i,j) = \frac{ \partial r_i }{ \partial \Beta_j } | \Beta
	virtual void evaluateJacobian( const FloatMatrix& beta, cholmod_triplet* J ) = 0;

};

#endif // SPARSE_ENERGY_H
