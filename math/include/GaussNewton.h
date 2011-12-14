#ifndef GAUSS_NEWTON_H
#define GAUSS_NEWTON_H

#include <memory>

#include "Energy.h"
#include "FloatMatrix.h"

class GaussNewton
{
public:

	// Initialize Gauss-Newton solver
	// pEnergy->numFunctions() >= pEnergy->numParameters()
	GaussNewton( std::shared_ptr< Energy > pEnergy, float epsilon = 1e-6 );

	FloatMatrix minimize( const FloatMatrix& guess, float* pEnergyFound = nullptr, int* pNumIterations = nullptr );

private:

	std::shared_ptr< Energy > m_pEnergy;
	float m_epsilon;
};

#endif // GAUSS_NEWTON_H
