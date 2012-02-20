#include "PARDISOSolver.h"

#include <cassert>

template<>
PARDISOSolver< float, true >::PARDISOSolver()
{
	int options = MKL_DSS_DEFAULTS;
	options += MKL_DSS_SINGLE_PRECISION;
	options += MKL_DSS_ZERO_BASED_INDEXING;
	int retval = dss_create( m_handle, options );
	assert( retval == MKL_DSS_SUCCESS );
}

template<>
PARDISOSolver< float, false >::PARDISOSolver()
{
	int options = MKL_DSS_DEFAULTS;
	options += MKL_DSS_SINGLE_PRECISION;
	int retval = dss_create( m_handle, options );
	assert( retval == MKL_DSS_SUCCESS );
}

template<>
PARDISOSolver< double, true >::PARDISOSolver()
{
	int options = MKL_DSS_DEFAULTS;
	options += MKL_DSS_ZERO_BASED_INDEXING;
	int retval = dss_create( m_handle, options );
	assert( retval == MKL_DSS_SUCCESS );
}

template<>
PARDISOSolver< double, false >::PARDISOSolver()
{
	int options = MKL_DSS_DEFAULTS;
	int retval = dss_create( m_handle, options );
	assert( retval == MKL_DSS_SUCCESS );
}

// virtual
template< typename valueType, bool zeroBased >
PARDISOSolver< valueType, zeroBased >::~PARDISOSolver()
{
	int options = MKL_DSS_DEFAULTS;
	dss_delete( m_handle, options );
}

template< typename valueType, bool zeroBased >
void PARDISOSolver< valueType, zeroBased >::analyzePattern( int m, int n, int* rowIndex, int* columns, int nNonZeroes )
{
	int structureOptions = MKL_DSS_SYMMETRIC;
	int retval = dss_define_structure( m_handle, structureOptions, rowIndex, m, n, columns, nNonZeroes );
	assert( retval == MKL_DSS_SUCCESS );

	int reorderOptions = MKL_DSS_DEFAULTS;
	//int reorderOptions = MKL_DSS_AUTO_ORDER;
	retval = dss_reorder( m_handle, reorderOptions, NULL );
	assert( retval == MKL_DSS_SUCCESS );
}

template< typename valueType, bool zeroBased >
void PARDISOSolver< valueType, zeroBased >::factorize( valueType* values )
{
	//int options = MKL_DSS_INDEFINITE;
	int options = MKL_DSS_POSITIVE_DEFINITE;
	int retval = dss_factor_real( m_handle, options, values );
	assert( retval == MKL_DSS_SUCCESS );
}

template< typename valueType, bool zeroBased >
void PARDISOSolver< valueType, zeroBased >::solve( valueType* rhs, valueType* solution )
{
	int options = MKL_DSS_DEFAULTS;
	int nRhs = 1;
	int retval = dss_solve_real( m_handle, options, rhs, nRhs, solution );
	assert( retval == MKL_DSS_SUCCESS );
}

// instantiate

template
PARDISOSolver< float, true >::~PARDISOSolver();

template
PARDISOSolver< float, false >::~PARDISOSolver();

template
PARDISOSolver< double, true >::~PARDISOSolver();

template
PARDISOSolver< double, false >::~PARDISOSolver();

template
void PARDISOSolver< float, true >::analyzePattern( int m, int n, int* rowIndex, int* columns, int nNonZeroes );

template
void PARDISOSolver< float, false >::analyzePattern( int m, int n, int* rowIndex, int* columns, int nNonZeroes );

template
void PARDISOSolver< double, true >::analyzePattern( int m, int n, int* rowIndex, int* columns, int nNonZeroes );

template
void PARDISOSolver< double, false >::analyzePattern( int m, int n, int* rowIndex, int* columns, int nNonZeroes );

template
void PARDISOSolver< float, true >::factorize( float* values );

template
void PARDISOSolver< float, false >::factorize( float* values );

template
void PARDISOSolver< double, true >::factorize( double* values );

template
void PARDISOSolver< double, false >::factorize( double* values );

template
void PARDISOSolver< float, true >::solve( float* rhs, float* solution );

template
void PARDISOSolver< float, false >::solve( float* rhs, float* solution );

template
void PARDISOSolver< double, true >::solve( double* rhs, double* solution );

template
void PARDISOSolver< double, false >::solve( double* rhs, double* solution );
