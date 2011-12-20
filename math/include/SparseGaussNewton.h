#ifndef SPARSE_GAUSS_NEWTON_H
#define SPARSE_GAUSS_NEWTON_H

#include <memory>

#include "SparseEnergy.h"
#include "FloatMatrix.h"

#include <cholmod.h>

class SparseGaussNewton
{
public:

	// Initialize Sparse Gauss-Newton solver
	// pEnergy->numFunctions() >= pEnergy->numParameters()
	//
	// Parameters:
	//   epsilon: minimize() will run until the residual has squared norm < epsilon
	//   or...
	//
	//   maxNumIterations n: minimize() will run for at most n iterations.
	//   Set to a negative number to ignore
	//
	SparseGaussNewton( std::shared_ptr< SparseEnergy > pEnergy, cholmod_common* pcc,
		int maxNumIterations = 100,
		float epsilon = 1e-6 );
	virtual ~SparseGaussNewton();

	int maxNumIterations() const;
	void setMaxNumIterations( int maxNumIterations );

	float epsilon() const;
	void setEpsilon( float epsilon );

	FloatMatrix minimize( float* pEnergyFound = nullptr, int* pNumIterations = nullptr );

private:

	std::shared_ptr< SparseEnergy > m_pEnergy;
	int m_maxNumIterations;
	float m_epsilon;

	cholmod_common* m_pcc;
	cholmod_triplet* m_J;	
	FloatMatrix m_prevBeta;
	FloatMatrix m_currBeta;
	FloatMatrix m_delta;
	FloatMatrix m_r;
	cholmod_dense* m_r2;
};

#endif // SPARSE_GAUSS_NEWTON_H
