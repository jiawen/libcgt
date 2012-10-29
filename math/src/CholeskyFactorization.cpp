#include "CholeskyFactorization.h"

#include <mkl.h>

CholeskyFactorization::CholeskyFactorization( const FloatMatrix& a, MatrixTriangle storedTriangle ) :

	m_storedTriangle( storedTriangle ),
	m_valid( false )
{
	int m = a.numRows();
	int n = a.numCols();

	if( m == n )
	{
		m_factor = a.extractTriangle( storedTriangle );
		char uplo = ( storedTriangle == LOWER ) ? 'L' : 'U';
		int info;
		
		spotrf( &uplo, &n, m_factor.data(), &n, &info );

		if( info < 0 )
		{
			fprintf( stderr, "CholeskyFactorization: Illegal parameter value: %d.\n", -info );
		}
		else if( info > 0 )
		{
			fprintf( stderr, "CholeskyFactorization: Matrix is not positive-definite.\n" );
		}
		else
		{
			m_valid = true;
		}
	}
}

FloatMatrix CholeskyFactorization::inverse() const
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

bool CholeskyFactorization::inverse( FloatMatrix& output ) const
{
	if( !isValid() )
	{
		return false;
	}

	char uplo = ( m_storedTriangle == LOWER ) ? 'L' : 'U';	
	int n = m_factor.numRows();

	output.copy( m_factor );

	int info;
	spotri( &uplo, &n, output.data(), &n, &info );
	if( info < 0 )
	{
		fprintf( stderr, "CholeskyFactorization::inverse(): Illegal parameter value.\n" );
		return false;
	}
	else if( info > 0 )
	{
		fprintf( stderr, "CholeskyFactorization::inverse(): A is singular.\n" );
		return false;
	}

	if( m_storedTriangle == LOWER )
	{
		output.copyTriangle( output, LOWER, UPPER, -1 );
	}
	else
	{
		output.copyTriangle( output, UPPER, LOWER, 1 );
	}

	return true;
}

FloatMatrix CholeskyFactorization::solve( const FloatMatrix& rhs ) const
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

bool CholeskyFactorization::solve( const FloatMatrix& rhs, FloatMatrix& solution ) const
{
	char uplo = ( m_storedTriangle == LOWER ) ? 'L' : 'U';	
	int n = m_factor.numRows();
	int ldaB = rhs.numRows();
	int nrhs = rhs.numCols();

	// make a copy of the right hand side
	solution.copy( rhs );

	int info;
	spotrs( &uplo, &n, &nrhs, m_factor.data(), &n, solution.data(), &ldaB, &info );

	if( info < 0 )
	{
		fprintf( stderr, "CholeskyFactorization::solve(): Illegal parameter value: %d\n", -info );
		return false;
	}
	return true;
}

MatrixTriangle CholeskyFactorization::storedTriangle() const
{
	return m_storedTriangle;
}

bool CholeskyFactorization::isValid() const
{
	return m_valid;
}

const FloatMatrix& CholeskyFactorization::factor() const
{
	return m_factor;
}

#if 0
FloatMatrix cholesky( const FloatMatrix& a )
{
	FloatMatrix l;

	int m = a.numRows();
	int n = a.numCols();

	if( m != n )
	{
		return l;
	}

	l.resize( n, n );

	for( int i = 0; i < a.numRows(); ++i )
	{
		for( int j = 0; j < i + 1; ++j )
		{
			float s = 0;

			for( int k = 0; k < j; ++k )
			{
				s += l( i, k ) * l( j, k );
			}

			if( i == j )
			{
				l( i, j ) = sqrt( a( i, i ) - s );
			}
			else
			{
				l( i, j ) = a( i, j ) / l( j, j ) - s;
			}
		}
	}

	return l;
}

// solution is just two triangular solves:
// Ax = b
// --> L Lt x = b
// Let y = Lt x
// then L y = b
// solving for y is a triangular solve
// Lt x = y
// solving for x is a triangular solve

#if 0
rank one update is easy:
function [L] = cholupdate(L,x)
	p = length(x);
x = x';
	for k=1:p
		r = sqrt(L(k,k)^2 + x(k)^2);
c = r / L(k, k);
s = x(k) / L(k, k);
L(k, k) = r;
L(k,k+1:p) = (L(k,k+1:p) + s*x(k+1:p)) / c;
x(k+1:p) = c*x(k+1:p) - s*L(k, k+1:p);
end
	end
#endif

#endif