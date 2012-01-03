#include "SparseGaussNewton.h"

#include <limits>
#include <QtGlobal>

#include <SuiteSparseQR.hpp>

void copyFloatMatrixToCholmodDense( const FloatMatrix& src, cholmod_dense* dst )
{
	double* dstArray = reinterpret_cast< double* >( dst->x );
	for( int k = 0; k < src.numElements(); ++k )
	{
		dstArray[k] = src[k];
	}
}

void copyCholmodDenseToFloatMatrix( cholmod_dense* src, FloatMatrix& dst )
{
	double* srcArray = reinterpret_cast< double* >( src->x );
	for( int k = 0; k < dst.numElements(); ++k )
	{
		dst[k] = srcArray[k];
	}
}

void saveVector( const FloatMatrix& x, QString filename )
{
	FILE* fp = fopen( qPrintable( filename ), "w" );
	for( int i = 0; i < x.numElements(); ++i )
	{
		fprintf( fp, "%f\n", x[i] );
	}
	fclose( fp );
}

#define TIMING 0
#define FACTORIZE 0
#define SPLIT_FACTORIZATION 0

#if TIMING
#include <time/StopWatch.h>
#endif


SparseGaussNewton::SparseGaussNewton( std::shared_ptr< SparseEnergy > pEnergy, cholmod_common* pcc,
	int maxNumIterations, float epsilon ) :

	m_maxNumIterations( maxNumIterations ),

	m_pcc( pcc ),
	m_J( nullptr ),
	m_pFactorization( nullptr ),

	m_r2( nullptr )

{
	setEpsilon( epsilon );
	setEnergy( pEnergy );	
}

SparseGaussNewton::~SparseGaussNewton()
{
	if( m_r2 != nullptr )
	{
		cholmod_l_free_dense( &m_r2, m_pcc );
	}
	if( m_pFactorization != nullptr )
	{
		SuiteSparseQR_free< double >( &m_pFactorization, m_pcc );
	}		
	if( m_J != nullptr )
	{
		cholmod_l_free_triplet( &m_J, m_pcc );
	}	
	m_pcc = nullptr;
}

void SparseGaussNewton::setEnergy( std::shared_ptr< SparseEnergy > pEnergy )
{
	printf( "resetting energy!\n" );

	m_pEnergy = pEnergy;

	int m = pEnergy->numFunctions();
	int n = pEnergy->numVariables();
	Q_ASSERT_X( m >= n, "Gauss Newton", "Number of functions (m) must be greater than the number of parameters (n)." );

	m_prevBeta.resize( n, 1 );
	m_currBeta.resize( n, 1 );
	m_delta.resize( n, 1 );
	m_r.resize( m, 1 );

	int nzMax = pEnergy->maxNumNonZeroes();

	// if m_r2 already exists
	if( m_r2 != nullptr )
	{
		// and the sizes don't match
		if( m_r2->nrow != m )
		{
			// then free it and re-allocate
			cholmod_l_free_dense( &m_r2, m_pcc );
			// TODO: use realloc instead?
			m_r2 = cholmod_l_allocate_dense( m, 1, m, CHOLMOD_REAL, m_pcc );
		}
	}
	else
	{
		m_r2 = cholmod_l_allocate_dense( m, 1, m, CHOLMOD_REAL, m_pcc );
	}

	if( m_J != nullptr )
	{
		if( m_J->nrow != m ||
			m_J->ncol != n ||
			m_J->nzmax != nzMax )
		{
			cholmod_l_free_triplet( &m_J, m_pcc );
			// TODO: use realloc instead?
			m_J = cholmod_l_allocate_triplet( m, n, nzMax, 0, CHOLMOD_REAL, m_pcc );	
		}
	}
	else
	{
		m_J = cholmod_l_allocate_triplet( m, n, nzMax, 0, CHOLMOD_REAL, m_pcc );	
	}

	if( m_pFactorization != nullptr )
	{
		SuiteSparseQR_free< double >( &m_pFactorization, m_pcc );
	}
}

uint SparseGaussNewton::maxNumIterations() const
{
	return m_maxNumIterations;
}

void SparseGaussNewton::setMaxNumIterations( uint maxNumIterations )
{
	m_maxNumIterations = maxNumIterations;
}

float SparseGaussNewton::epsilon() const
{
	return m_epsilon;
}

void SparseGaussNewton::setEpsilon( float epsilon )
{
	m_epsilon = epsilon;
	m_sqrtEpsilon = sqrtf( epsilon );
}

const FloatMatrix& SparseGaussNewton::minimize( float* pEnergyFound, int* pNumIterations )
{
#if TIMING
	float tR = 0;
	float tCopy = 0;
	float tConvert = 0;
#if FACTORIZE
	float tFactorize = 0;
	float tQMult = 0;
	float tSolve = 0;
#else
	float tQR = 0;
#endif
	StopWatch sw;
#endif
	
	m_pEnergy->evaluateInitialGuess( m_currBeta );

#if TIMING
	sw.reset();
#endif
	m_J->nnz = 0;
	m_pEnergy->evaluateResidualAndJacobian( m_currBeta, m_r, m_J );
#if TIMING
	tR += sw.millisecondsElapsed();
	sw.reset();
#endif
	copyFloatMatrixToCholmodDense( m_r, m_r2 );
#if TIMING
	tCopy += sw.millisecondsElapsed();
#endif

	float prevEnergy = FLT_MAX;
	float currEnergy = FloatMatrix::dot( m_r, m_r );
	float deltaEnergy = fabs( currEnergy - prevEnergy );

	bool deltaEnergyConverged;
	bool deltaBetaConverged;
	bool converged = false;

	//printf( "initial energy = %f\n", currEnergy );	

	// check for convergence
	int nIterations = 0;
	while( ( nIterations < m_maxNumIterations ) &&
		!converged )
	{
		// not converged
		prevEnergy = currEnergy;
		//m_prevBeta = m_currBeta;

#if 0
		if( nIterations == 0 )
		{
			//printf( "prev beta = %s\n", qPrintable( m_prevBeta.toString() ) );
			//printf( "curr beta = %s\n", qPrintable( m_currBeta.toString() ) );
			//printf( "curr residual =\n%s\n", qPrintable( m_r.toString() ) );
			saveVector( m_r, "c:/tmp/r0.txt" );
		}
#endif

#if TIMING
		sw.reset();
#endif
		cholmod_sparse* jSparse = cholmod_l_triplet_to_sparse( m_J, m_J->nnz, m_pcc );		
#if TIMING
		tConvert += sw.millisecondsElapsed();
		sw.reset();
#endif
		
#if FACTORIZE

#if SPLIT_FACTORIZATION // symbolic then numeric
		if( m_pFactorization == nullptr )
		{
			m_pFactorization = SuiteSparseQR_symbolic< double >( SPQR_ORDERING_DEFAULT, false, jSparse, m_pcc );
		}
		SuiteSparseQR_numeric< double >( SPQR_DEFAULT_TOL, jSparse, m_pFactorization, m_pcc );
#else
		// numeric directly
		if( m_pFactorization == nullptr )
		{
			//printf( "factorization is null, factorizing...\n" );
			//m_pFactorization = SuiteSparseQR_factorize< double >( SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL, jSparse, m_pcc );
			//m_pFactorization = SuiteSparseQR_factorize< double >( SPQR_ORDERING_DEFAULT, 1e-3, jSparse, m_pcc );
			//m_pFactorization = SuiteSparseQR_factorize< double >( SPQR_ORDERING_CHOLMOD, 1e-3, jSparse, m_pcc );
			m_pFactorization = SuiteSparseQR_factorize< double >( SPQR_ORDERING_CHOLMOD, SPQR_DEFAULT_TOL, jSparse, m_pcc );
			//m_pFactorization = SuiteSparseQR_factorize< double >( SPQR_ORDERING_METIS, 1e-3, jSparse, m_pcc );
			//m_pFactorization = SuiteSparseQR_factorize< double >( SPQR_ORDERING_BEST, 1e-3, jSparse, m_pcc );
			//m_pFactorization = SuiteSparseQR_factorize< double >( SPQR_ORDERING_COLAMD, 1e-3, jSparse, m_pcc );
			//m_pFactorization = SuiteSparseQR_factorize< double >( SPQR_ORDERING_AMD, 1e-3, jSparse, m_pcc );			
		}
		else
		{
			//printf( "factorization exists, reusing...\n" );
			SuiteSparseQR_numeric< double >( SPQR_DEFAULT_TOL, jSparse, m_pFactorization, m_pcc );
			//SuiteSparseQR_numeric< double >( 1e-3, jSparse, m_pFactorization, m_pcc );
		}
#endif

#if TIMING
		tFactorize += sw.millisecondsElapsed();
		sw.reset();
#endif

		auto y = SuiteSparseQR_qmult< double >( SPQR_QTX, m_pFactorization, m_r2, m_pcc );

#if TIMING
		tQMult += sw.millisecondsElapsed();
		sw.reset();
#endif
		auto delta = SuiteSparseQR_solve< double >( SPQR_RETX_EQUALS_B, m_pFactorization, y, m_pcc );
#if TIMING
		tSolve += sw.millisecondsElapsed();
#endif

#else
		auto delta = SuiteSparseQR< double >( jSparse, m_r2, m_pcc );
#if TIMING
		tQR += sw.millisecondsElapsed();
#endif

#endif

#if TIMING
		sw.reset();
#endif
		copyCholmodDenseToFloatMatrix( delta, m_delta );
#if TIMING
		tCopy += sw.millisecondsElapsed();
#endif

		m_J->nnz = 0; // reset sparse jacobian
		cholmod_l_free_dense( &delta, m_pcc );
#if FACTORIZE
		cholmod_l_free_dense( &y, m_pcc );
#endif
		cholmod_l_free_sparse( &jSparse, m_pcc );
		
		//m_currBeta = m_prevBeta - m_delta;
		m_currBeta -= m_delta;

#if 0
		if( nIterations == 0 )
		{
			saveVector( m_delta, "c:/tmp/delta_0.txt" );
			saveVector( m_prevBeta, "c:/tmp/m_prevBeta.txt" );
			saveVector( m_currBeta, "c:/tmp/m_currBeta.txt" );
			//printf( "initial delta =\n%s\n", qPrintable( m_delta.toString() ) );
			//printf( "beta[1] =\n%s\n", qPrintable( m_currBeta.toString() ) );
		}
#endif

		// update energy
#if TIMING
		sw.reset();
#endif
		m_pEnergy->evaluateResidualAndJacobian( m_currBeta, m_r, m_J );
#if TIMING
		tR += sw.millisecondsElapsed();
		sw.reset();
#endif
		copyFloatMatrixToCholmodDense( m_r, m_r2 );
#if TIMING
		tCopy += sw.millisecondsElapsed();
#endif

		currEnergy = FloatMatrix::dot( m_r, m_r );
		deltaEnergy = fabs( currEnergy - prevEnergy );

		deltaEnergyConverged = ( deltaEnergy < m_epsilon * ( 1 + currEnergy ) );

#if 0
		//float deltaBetaMax = m_delta.maximum();
		//deltaBetaConverged = ( deltaBetaMax < m_sqrtEpsilon * ( 1 + deltaBetaMax ) );		
		converged = deltaEnergyConverged && deltaBetaConverged;
#else
		converged = deltaEnergyConverged;

		//printf( "k = %d, E[k] = %f, |deltaE| = %f, eps * ( 1 + E[k] ) = %f, converged = %d\n",
		//	nIterations, currEnergy, deltaEnergy, m_epsilon * ( 1 + currEnergy ), (int)deltaEnergyConverged );

#endif
		++nIterations;
	}

#if TIMING
#if FACTORIZE
	printf( "timing breakdown:\ntR = %f, tCopy = %f, tConvert = %f, tFactorize = %f, tQMult = %f, tSolve = %f\n",
		tR, tCopy, tConvert, tFactorize, tQMult, tSolve );
#else
	printf( "timing breakdown:\ntR = %f, tCopy = %f, tConvert = %f, tQR = %f\n",
		tR, tCopy, tConvert, tQR );
#endif
#endif

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
