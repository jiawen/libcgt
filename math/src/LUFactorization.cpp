#include "LUFactorization.h"

#include <algorithm>
#include <numeric>

#include <mkl.h>

LUFactorization::LUFactorization( const FloatMatrix& a ) :

	m_nRowsA( a.numRows() ),
	m_nColsA( a.numCols() ),

	m_valid( false ),
	m_y( a ),
	m_ipiv( std::min( a.numRows(), a.numCols() ) )
{
	int m = a.numRows();
	int n = a.numCols();

	int info;
	sgetrf( &m, &n, m_y.data(), &m, m_ipiv.data(), &info );
	if( info < 0 )
	{
		fprintf( stderr, "LUFactorization: Illegal parameter value.\n" );
	}
	else if( info > 0 )
	{
		fprintf( stderr, "LUFactorization: Matrix is singular.\n" );
	}
	else
	{
		m_valid = true;
	}
}

bool LUFactorization::isValid() const
{
	return m_valid;
}

FloatMatrix LUFactorization::inverse() const
{
	FloatMatrix output;
	if( inverse( output ) )
	{
		return output;
	}
	else
	{
		return FloatMatrix();
	}
}

bool LUFactorization::inverse( FloatMatrix& output ) const
{
	if( !isValid() )
	{
		return false;
	}

	int m = m_nRowsA;
	int n = m_nColsA;

	output.copy( m_y );

	// ask for how much work space is needed		
	float workQuery;
	int lwork = -1;
	int info;

	sgetri( &n, output.data(), &m, m_ipiv.data(), &workQuery, &lwork, &info );
	if( info < 0 )
	{
		fprintf( stderr, "LUFactorization::inverse(): Illegal parameter value.\n" );
		return false;
	}

	lwork = static_cast< int >( workQuery );
	std::vector< float > work( lwork );

	// solve
	sgetri( &n, output.data(), &m, m_ipiv.data(), work.data(), &lwork, &info );
	if( info < 0 )
	{
		fprintf( stderr, "LUFactorization::inverse(): Illegal parameter value: %d.\n", -info );
		return false;
	}		
	if( info > 0 )
	{
		fprintf( stderr, "LUFactorization::inverse(): A is singular.\n" );
		return false;
	}
	return true;
}

FloatMatrix LUFactorization::solve( const FloatMatrix& rhs ) const
{
	FloatMatrix solution;
	if( solve( rhs, solution ) )
	{
		return solution;
	}
	else
	{
		return FloatMatrix();
	}
}

bool LUFactorization::solve( const FloatMatrix& rhs, FloatMatrix& solution ) const
{
	if( m_nRowsA != m_nColsA )
	{
		return false;
	}

	char trans = 'N';
	int n = m_nRowsA;
	int ldaB = rhs.numRows();
	int nrhs = rhs.numCols();

	// make a copy of the right hand side
	solution.copy( rhs );

	int info;
	sgetrs( &trans, &n, &nrhs, m_y.data(), &n, m_ipiv.data(), solution.data(), &ldaB, &info );
	
	if( info < 0 )
	{
		fprintf( stderr, "LUFactorization::solve(): Illegal parameter value: %d\n", -info );
		return false;
	}
	return true;
}

const FloatMatrix& LUFactorization::y() const
{
	return m_y;
}

const std::vector< int >& LUFactorization::ipiv() const
{
	return m_ipiv;
}

FloatMatrix LUFactorization::l() const
{
	FloatMatrix lower = m_y.extractTriangle( LOWER, -1 );

	int n = static_cast< int >( m_ipiv.size() );
	for( int i = 0; i < n; ++i )
	{
		lower( i, i ) = 1;
	}
	
	return lower;
}

FloatMatrix LUFactorization::u() const
{
	return m_y.extractTriangle( UPPER );
}

std::vector< int > LUFactorization::pivotVector() const
{
	int n = static_cast< int >( m_ipiv.size() );
	
	std::vector< int > pivot( n );
	std::iota( pivot.begin(), pivot.end(), 0 );

	for( int i = 0; i < n; ++i )
	{
		int p = m_ipiv[ i ] - 1;
		std::swap( pivot[ i ], pivot[ p ] );
	}

	return pivot;
}

FloatMatrix LUFactorization::permutationMatrix() const
{
	int n = static_cast< int >( m_ipiv.size() );

	std::vector< int > pivot = pivotVector();
	FloatMatrix perm( n, n );

	for( int i = 0; i < n; ++i )
	{
		perm( i, pivot[i] ) = 1;
	}

	return perm;
}

#if 0

// solution is just two triangular solves:
// Ax = b and PA = LU
// then LUx = Pb
// let y = Ux
// then Ly = Pb
// solving for y is a triangular solve
// Ux = y
// solving for x is a triangular solve

#endif