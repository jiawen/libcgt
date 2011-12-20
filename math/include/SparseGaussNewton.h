#ifndef SPARSE_GAUSS_NEWTON_H
#define SPARSE_GAUSS_NEWTON_H

#include <memory>

#include "SparseEnergy.h"
#include "FloatMatrix.h"

#include <cholmod.h>

class SparseGaussNewton
{
public:

	// Initialize Gauss-Newton solver
	// pEnergy->numFunctions() >= pEnergy->numParameters()
	SparseGaussNewton( std::shared_ptr< SparseEnergy > pEnergy, cholmod_common* pcc, int maxNumIterations = 100, float epsilon = 1e-6 );
	virtual ~SparseGaussNewton();

	FloatMatrix minimize( const FloatMatrix& guess, float* pEnergyFound = nullptr, int* pNumIterations = nullptr );

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
