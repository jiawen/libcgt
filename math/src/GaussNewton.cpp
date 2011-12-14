#include "GaussNewton.h"

#include <limits>
#include <QtGlobal>

#include "LinearLeastSquaresSolvers.h"

GaussNewton::GaussNewton( std::shared_ptr< Energy > pEnergy, float epsilon ) :

	m_pEnergy( pEnergy ),
	m_epsilon( epsilon )

{
	int m = pEnergy->numFunctions();
	int n = pEnergy->numVariables();

	Q_ASSERT_X( m >= n, "Gauss Newton", "Number of functions (m) must be greater than the number of parameters (n)." );
}

FloatMatrix GaussNewton::minimize( const FloatMatrix& guess, float* pEnergyFound, int* pNumIterations )
{
	FloatMatrix J( m_pEnergy->numFunctions(), m_pEnergy->numVariables() );
	FloatMatrix prevBeta( guess );
	FloatMatrix currBeta( guess );
	FloatMatrix delta( guess.numRows(), guess.numCols() );
	FloatMatrix r( m_pEnergy->numFunctions(), 1 );

	m_pEnergy->evaluateResidual( currBeta, r );

	float prevEnergy = FLT_MAX;
	float currEnergy = FloatMatrix::dot( r, r );
	float deltaEnergy = fabs( currEnergy - prevEnergy );
	
	// check for convergence
	int nIterations = 0;
	while( deltaEnergy > m_epsilon * ( 1 + currEnergy ) )
	{
		// not converged
		prevEnergy = currEnergy;
		prevBeta = currBeta;

		// take a step:			
		m_pEnergy->evaluateJacobian( currBeta, J );

		//J.print( "J = " );

		// TODO: OPTIMIZE: -r?
		LinearLeastSquaresSolvers::QRFullRank( J, -r, delta );
		// TODO: OPTIMIZE: do it in place
		currBeta = prevBeta + delta;

		// update energy
		m_pEnergy->evaluateResidual( currBeta, r );
		currEnergy = FloatMatrix::dot( r, r );
		deltaEnergy = fabs( currEnergy - prevEnergy );
		++nIterations;
	}

	if( pEnergyFound != nullptr )
	{
		*pEnergyFound = currEnergy;
	}

	if( pNumIterations != nullptr )
	{
		*pNumIterations = nIterations;
	}
	return currBeta;
}
