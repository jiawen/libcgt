#include "SparseGaussNewton.h"

#include <limits>
#include <QtGlobal>

#include <SuiteSparseQR.hpp>

SparseGaussNewton::SparseGaussNewton( std::shared_ptr< SparseEnergy > pEnergy, cholmod_common* pcc,
	int maxNumIterations, float epsilon ) :

	m_pEnergy( pEnergy ),
	m_maxNumIterations( maxNumIterations ),
	m_epsilon( epsilon ),

	m_pcc( pcc )

{
	int m = pEnergy->numFunctions();
	int n = pEnergy->numVariables();
	int nzMax = pEnergy->maxNumNonZeroes();

	m_J = cholmod_l_allocate_triplet( m, n, nzMax, 0, CHOLMOD_REAL, m_pcc );
	m_prevBeta.resize( n, 1 );
	m_currBeta.resize( n, 1 );
	m_delta.resize( n, 1 );
	m_r.resize( m, 1 );

	m_r2 = cholmod_l_allocate_dense( m, 1, m, CHOLMOD_REAL, m_pcc );

	Q_ASSERT_X( m >= n, "Gauss Newton", "Number of functions (m) must be greater than the number of parameters (n)." );
}

SparseGaussNewton::~SparseGaussNewton()
{
	cholmod_l_free_dense( &m_r2, m_pcc );
	cholmod_l_free_triplet( &m_J, m_pcc );
	m_pcc = nullptr;
}

void copyFloatMatrixToCholmodDense( const FloatMatrix& src, cholmod_dense* dst )
{
	double* dstArray = reinterpret_cast< double* >( dst->x );
	for( int k = 0; k < src.numElements(); ++k )
	{
		dstArray[k] = src[k];
	}
}

void copyCholmodDenseToFloatMatrix( cholmod_dense* src, FloatMatrix& dst)
{
	double* srcArray = reinterpret_cast< double* >( src->x );
	for( int k = 0; k < dst.numElements(); ++k )
	{
		dst[k] = srcArray[k];
	}
}

FloatMatrix SparseGaussNewton::minimize( const FloatMatrix& guess, float* pEnergyFound, int* pNumIterations )
{
	m_currBeta.copy( guess );
	m_pEnergy->evaluateResidual( m_currBeta, m_r );
	copyFloatMatrixToCholmodDense( m_r, m_r2 );

	float prevEnergy = FLT_MAX;
	float currEnergy = FloatMatrix::dot( m_r, m_r );
	float deltaEnergy = fabs( currEnergy - prevEnergy );
	

	// check for convergence
	int nIterations = 0;
	while( deltaEnergy > m_epsilon * ( 1 + currEnergy ) &&
		nIterations < m_maxNumIterations )
	{
		// not converged
		prevEnergy = currEnergy;
		m_prevBeta = m_currBeta;

		// take a step:
		m_pEnergy->evaluateJacobian( m_currBeta, m_J );
		//J.print( "J = " );

		cholmod_sparse* jSparse = cholmod_l_triplet_to_sparse( m_J, m_J->nnz, m_pcc );		

		auto delta = SuiteSparseQR< double >( jSparse, m_r2, m_pcc );

		copyCholmodDenseToFloatMatrix( delta, m_delta );
		
		m_J->nnz = 0; // reset sparse jacobian
		cholmod_l_free_dense( &delta, m_pcc );
		cholmod_l_free_sparse( &jSparse, m_pcc );

		// TODO: OPTIMIZE: do it in place
		m_currBeta = m_prevBeta - m_delta;

		// update energy
		m_pEnergy->evaluateResidual( m_currBeta, m_r );
		copyFloatMatrixToCholmodDense( m_r, m_r2 );

		currEnergy = FloatMatrix::dot( m_r, m_r );
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
	return m_currBeta;
}
