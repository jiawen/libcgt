#include "FloatMatrix.h"

#include <mkl.h>
#include <mkl_blas.h>

#include <cassert>
#include <cstdarg>

#include <algorithm>

#include "LUFactorization.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
FloatMatrix FloatMatrix::zeroes( int m, int n )
{
	return FloatMatrix( m, n );
}

// static
FloatMatrix FloatMatrix::ones( int m, int n )
{
	return FloatMatrix( m, n, 1.0f );
}

FloatMatrix::FloatMatrix() :

	m_nRows( 0 ),
	m_nCols( 0 )

{

}

FloatMatrix::FloatMatrix( int nRows, int nCols, float fillValue ) :

	m_nRows( nRows ),
	m_nCols( nCols ),
	m_data( nRows * nCols, fillValue )

{

}

FloatMatrix::FloatMatrix( const FloatMatrix& copy ) :

	m_nRows( copy.m_nRows ),
	m_nCols( copy.m_nCols ),
	m_data( copy.m_data )

{

}

FloatMatrix::FloatMatrix( FloatMatrix&& move )
{
	m_data = std::move( move.m_data );
	m_nRows = move.m_nRows;
	m_nCols = move.m_nCols;

	move.m_nRows = 0;
	move.m_nCols = 0;
}

// virtual
FloatMatrix::~FloatMatrix()
{

}

FloatMatrix& FloatMatrix::operator = ( const FloatMatrix& copy )
{
	if( this != &copy )
	{
		m_nRows = copy.m_nRows;
		m_nCols = copy.m_nCols;
		m_data = copy.m_data;
	}
	return *this;
}

FloatMatrix& FloatMatrix::operator = ( FloatMatrix&& move )
{
	if( this != &move )
	{
		m_data = std::move( move.m_data );
		m_nRows = move.m_nRows;
		m_nCols = move.m_nCols;

		move.m_nRows = 0;
		move.m_nCols = 0;
	}
	return *this;
}

int FloatMatrix::numRows() const
{
	return m_nRows;
}

int FloatMatrix::numCols() const
{
	return m_nCols;
}

int FloatMatrix::numElements() const
{
	return static_cast< int >( m_data.size() );
}

Vector2i FloatMatrix::indexToSubscript( int idx ) const
{
	return Vector2i( idx / m_nRows, idx % m_nRows );
}

int FloatMatrix::subscriptToIndex( int i, int j ) const
{
	return( j * m_nRows + i );
}

void FloatMatrix::resize( int nRows, int nCols )
{
	if( nRows == m_nRows && nCols == m_nCols )
	{
		return;
	}

	m_nRows = nRows;
	m_nCols = nCols;
	m_data.resize( nRows * nCols, 0 );
}

bool FloatMatrix::reshape( int nRows, int nCols )
{
	if( nRows * nCols != numElements() )
	{
		return false;
	}

	m_nRows = nRows;
	m_nCols = nCols;
	return true;
}

bool FloatMatrix::isNull() const
{
	return( m_nRows == 0 || m_nCols == 0 );
}

void FloatMatrix::fill( float d )
{
	for( int i = 0; i < m_data.size(); ++i )
	{
		m_data[ i ] = d;
	}	
}

float& FloatMatrix::operator () ( int i, int j )
{
	return( m_data[ j * m_nRows + i ] );
}

const float& FloatMatrix::operator () ( int i, int j ) const
{
	return( m_data[ j * m_nRows + i ] );
}

float& FloatMatrix::operator [] ( int k )
{
	return m_data[ k ];
}

const float& FloatMatrix::operator [] ( int k ) const
{
	return m_data[ k ];
}

void FloatMatrix::copy( const FloatMatrix& m )
{
	resize( m.m_nRows, m.m_nCols );
	m_data = m.m_data;
}

void FloatMatrix::assign( const FloatMatrix& m, int i0, int j0, int i1, int j1, int nRows, int nCols )
{
	assert( i0 >= 0 );
	assert( i0 < numRows() );
	assert( j0 >= 0 );
	assert( j0 < numCols() );

	assert( i1 >= 0 );
	assert( i1 < m.numRows() );
	assert( j1 >= 0 );
	assert( j1 < m.numCols() );

	if( nRows == 0 )
	{
		nRows = m.numRows() - i1;
	}
	if( nCols == 0 )
	{
		nCols = m.numCols() - j1;
	}	

	assert( i0 + nRows <= numRows() );
	assert( j0 + nCols <= numCols() );
	assert( i1 + nRows <= m.numRows() );
	assert( j1 + nCols <= m.numCols() );

	for( int i = 0; i < nRows; ++i )
	{
		for( int j = 0; j < nCols; ++j )
		{
			(*this)( i0 + i, j0 + j ) = m( i1 + i, j1 + j );
		}
	}
}

const float* FloatMatrix::data() const
{
	return m_data.data();
}

float* FloatMatrix::data()
{
	return m_data.data();
}

FloatMatrix FloatMatrix::solve( const FloatMatrix& rhs ) const
{
	bool b;
	return solve( rhs, b );
}

FloatMatrix FloatMatrix::solve( const FloatMatrix& rhs, bool& succeeded ) const
{
	// check that this matrix is square
	int m = m_nRows;
	int n = m_nCols;

	if( m != n )
	{
		fprintf( stderr, "FloatMatrix::solve requires the matrix to be square\n" );
		return FloatMatrix();
	}

	int nRHS = rhs.numCols();
	if( rhs.numRows() != n )
	{
		fprintf( stderr, "FloatMatrix::solve: dimension mismatch.  A = %d x %d, rhs = %d x %d\n",
			m, n, rhs.numRows(), nRHS );
		return FloatMatrix();
	}

	// make a copy of this and the right hand side
	FloatMatrix A( *this );
	FloatMatrix B( rhs );

	std::vector< int > ipiv( n );
	int info;

	sgesv( &n, &nRHS, A.data(), &m, ipiv.data(), B.data(), &n, &info );

	succeeded = ( info == 0 );

	if( info < 0 )
	{
		fprintf( stderr, "FloatMatrix::solve(): Illegal parameter value: %d\n", -info );
		return FloatMatrix();
	}
	else if( info > 0 )
	{
		fprintf( stderr, "FloatMatrix::solve(): Matrix is singular.\n" );
		return FloatMatrix();
	}

	return B;
}

FloatMatrix FloatMatrix::solveSPD( const FloatMatrix& rhs, MatrixTriangle storedTriangle ) const
{
	bool succeeded;
	return solveSPD( rhs, succeeded, storedTriangle );
}

FloatMatrix FloatMatrix::solveSPD( const FloatMatrix& rhs, bool& succeeded, MatrixTriangle storedTriangle ) const
{
	// check that this matrix is square
	int m = m_nRows;
	int n = m_nCols;

	if( m != n )
	{
		fprintf( stderr, "FloatMatrix::solve requires the matrix to be square\n" );
		return FloatMatrix();
	}

	int nRHS = rhs.numCols();
	if( rhs.numRows() != n )
	{
		fprintf( stderr, "FloatMatrix::solve: dimension mismatch.  A = %d x %d, rhs = %d x %d\n",
			m, n, rhs.numRows(), nRHS );
		return FloatMatrix();
	}

	// make a copy of this and the right hand side
	FloatMatrix A( *this );
	FloatMatrix B( rhs );

	char uplo = ( storedTriangle == LOWER ) ? 'L' : 'U';
	int info;

	sposv( &uplo, &n, &nRHS, A.data(), &m, B.data(), &m, &info );
	
	succeeded = ( info == 0 );

	if( info < 0 )
	{
		fprintf( stderr, "FloatMatrix::solveSPD(): Illegal parameter value: %d\n", -info );		
		return FloatMatrix();
	}
	else if( info > 0 )
	{
		fprintf( stderr, "FloatMatrix::solveSPD(): Matrix is not positive definite.\n" );
		return FloatMatrix();
	}

	return B;
}

FloatMatrix FloatMatrix::inverted( bool* pSucceeded ) const
{
	FloatMatrix inv;
	LUFactorization lu( *this );
	bool succeeded = lu.inverse( inv );

	if( pSucceeded != nullptr )
	{
		*pSucceeded = succeeded;
	}

	return inv;
}

bool FloatMatrix::inverted( FloatMatrix& inv ) const
{
	LUFactorization lu( *this );
	return lu.inverse( inv );
}

void FloatMatrix::transposed( FloatMatrix& t ) const
{
	int M = m_nRows;
	int N = m_nCols;

	t.resize( N, M );

	for( int i = 0; i < M; ++i )
	{
		for( int j = 0; j < N; ++j )
		{
			t( j, i ) = ( *this )( i, j );
		}
	}
}

FloatMatrix FloatMatrix::transposed() const
{
	FloatMatrix t( m_nCols, m_nRows );
	transposed( t );
	return t;
}

float FloatMatrix::frobeniusNorm() const
{
	return norm( 'f' );
}

float FloatMatrix::frobeniusNormSquared() const
{
	int m = numElements();
	int inc = 1;
	return sdot( &m, data(), &inc, data(), &inc );
}

// returns the maximum row sum
float FloatMatrix::infinityNorm() const
{
	char whichNorm = 'i';
	int m = numRows();
	int n = numCols();

	std::vector< float > work( m );

	return slange
	(
		&whichNorm,
		&m, &n,
		data(),
		&m,
		work.data()
	);
}

// returns the maximum column sum
float FloatMatrix::l1Norm() const
{
	return norm( 'o' );
}

// Returns the element with largest absolute value
float FloatMatrix::maximumNorm() const
{
	return norm( 'm' );
}

FloatMatrix& FloatMatrix::operator += ( const FloatMatrix& x )
{
	scaledAdd( 1.0f, x, *this );
	return( *this );
}

FloatMatrix& FloatMatrix::operator -= ( const FloatMatrix& x )
{
	scaledAdd( -1.0f, x, *this );
	return( *this );
}

// static
float FloatMatrix::dot( const FloatMatrix& a, const FloatMatrix& b )
{
	assert( a.numRows() == 1 || a.numCols() == 1 );
	assert( b.numRows() == 1 || b.numCols() == 1 );
	assert( a.numElements() == b.numElements() );

	int m = a.numElements();
	int inc = 1;
	return sdot( &m, a.data(), &inc, b.data(), &inc );
}

// static
void FloatMatrix::add( const FloatMatrix& a, const FloatMatrix& b, FloatMatrix& c )
{
	assert( a.numRows() == b.numRows() );
	assert( a.numCols() == b.numCols() );

	// TODO: is this any faster?
	// copy c <-- b
	// c = b;
	// scaledAdd( 1, a, c );

	c.resize( a.numRows(), a.numCols() );

	for( int k = 0; k < a.numElements(); ++k )
	{
		c[k] = a[k] + b[k];
	}
}

// static
void FloatMatrix::subtract( const FloatMatrix& a, const FloatMatrix& b, FloatMatrix& c )
{
	assert( a.numRows() == b.numRows() );
	assert( a.numCols() == b.numCols() );

	c.resize( a.numRows(), a.numCols() );

	// TODO: use MKL
	for( int k = 0; k < a.numElements(); ++k )
	{
		c[k] = a[k] - b[k];
	}
}

// static
void FloatMatrix::multiply( const FloatMatrix& a, const FloatMatrix& b, FloatMatrix& c )
{
	assert( a.numCols() == b.numRows() );

	c.resize( a.numRows(), b.numCols() );

	scaledMultiplyAdd( 1, a, b, 0, c );
}

// static
void FloatMatrix::scaledAdd( float alpha, const FloatMatrix& x, FloatMatrix& y )
{	
	int n = x.numElements();
	assert( n == y.numElements() );
	int inc = 1;

	saxpy( &n, &alpha, x.data(), &inc, y.data(), &inc );
}

// static
void FloatMatrix::scaledMultiplyAdd( float alpha, const FloatMatrix& a, const FloatMatrix& b, float beta, FloatMatrix& c )
{
	int m = a.numRows();
	int n = a.numCols();
	
	assert( n = b.numRows() );
	int p = b.numCols();

	c.resize( m, p );

	char transa = 'n';
	char transb = 'n';
	
	sgemm
	(
		&transa, &transb,
		&m, &p, &n,
		&alpha,
		a.data(), &m,
		b.data(), &n,
		&beta,
		c.data(), &m
	);
}

#if 0
void FloatMatrix::eigenvalueDecomposition( QVector< QVector< float > >* eigen_vector,
									   QVector< float >* eigen_value )
{
	// TODO: assert matrix is square

	int N = m_nRows;
	float* gsl_proxy = new float[ N * N ];

	eigen_vector->resize(N);
	for( int i = 0; i < N; ++i )
	{
		( *eigen_vector )[ i ].resize( N );
	}

	eigen_value->resize(N);

	for( int i=0;i<N;i++){
		for( int j=0;j<N;j++){
			gsl_proxy[i*N+j] = ( *this )( i, j );
		}
	}

	// TODO: don't alloc, etc
	// TODO: MKL?
	gsl_matrix_view gsl_mat      = gsl_matrix_view_array(gsl_proxy,N,N);
	gsl_vector* gsl_eigen_value  = gsl_vector_alloc(N);
	gsl_matrix* gsl_eigen_vector = gsl_matrix_alloc(N,N);
	gsl_eigen_symmv_workspace* w = gsl_eigen_symmv_alloc(N);

	gsl_eigen_symmv(&gsl_mat.matrix,gsl_eigen_value,gsl_eigen_vector,w);
	gsl_eigen_symmv_sort(gsl_eigen_value,gsl_eigen_vector,GSL_EIGEN_SORT_ABS_ASC);

	for( int i=0;i<N;i++){

		(*eigen_value)[i] = gsl_vector_get(gsl_eigen_value,i);

		for( int j=0;j<N;j++){
			(*eigen_vector)[i][j] = gsl_matrix_get(gsl_eigen_vector,j,i);
		}
	}

	gsl_matrix_free(gsl_eigen_vector);
	gsl_vector_free(gsl_eigen_value);
	gsl_eigen_symmv_free(w);   
	delete[] gsl_proxy;
}

// static
void FloatMatrix::homography( QVector< Vector3f > from,
						  QVector< Vector3f > to, FloatMatrix& output )
{
	output.resize( 3, 3 );
	FloatMatrix m( 8, 9 );

	for( int i = 0; i < 4; ++i )
	{
		m(2*i,0) = from[i].x() * to[i].z();
		m(2*i,1) = from[i].y() * to[i].z();
		m(2*i,2) = from[i].z() * to[i].z();
		m(2*i,6) = -from[i].x() * to[i].x();
		m(2*i,7) = -from[i].y() * to[i].x();
		m(2*i,8) = -from[i].z() * to[i].x();

		m(2*i+1,3) = from[i].x() * to[i].z();
		m(2*i+1,4) = from[i].y() * to[i].z();
		m(2*i+1,5) = from[i].z() * to[i].z();
		m(2*i+1,6) = -from[i].x() * to[i].y();
		m(2*i+1,7) = -from[i].y() * to[i].y();
		m(2*i+1,8) = -from[i].z() * to[i].y();
	}

	QVector< QVector< float > > eigenvectors;
	QVector< float > eigenvalues;

	FloatMatrix mt( 9, 8 );
	m.transpose( mt );

	FloatMatrix mtm( 9, 9 );
	FloatMatrix::times( mt, m, mtm );

	mtm.eigenvalueDecomposition( &eigenvectors, &eigenvalues );

	for( int i=0;i<3;i++){
		for( int j=0;j<3;j++){
			output( i, j ) = eigenvectors[0][3*i+j];
		}
	}
}
#endif

float FloatMatrix::minimum() const
{
	return *( std::min_element( m_data.begin(), m_data.end() ) );
}

float FloatMatrix::maximum() const
{
	return *( std::max_element( m_data.begin(), m_data.end() ) );
}

bool FloatMatrix::loadTXT( QString filename )
{
	FILE* fp = fopen( qPrintable( filename ), "r" );
	bool succeeded = ( fp != NULL );

	if( succeeded )
	{
		int m = -1;
		int n = -1;
		fscanf( fp, "%d %d\n", &m, &n );
		
		succeeded = ( m >= 0 && m >= 0 );
		if( succeeded )
		{
			m_nRows = m;
			m_nCols = n;
			m_data.resize( m * n );

			for( size_t k = 0; k < m * n; ++k )
			{
				fscanf( fp, "%f\n", &( m_data[ k ] ) );
			}

			fclose( fp );
		}
	}

	return succeeded;
}

bool FloatMatrix::saveTXT( QString filename )
{
	FILE* fp = fopen( qPrintable( filename ), "w" );
	bool succeeded = ( fp != NULL );

	if( succeeded )
	{
		fprintf( fp, "%d %d\n", m_nRows, m_nCols );
		for( size_t k = 0; k < m_data.size(); ++k )
		{
			fprintf( fp, "%f\n", m_data[ k ] );
		}

		fclose( fp );
	}
	
	return succeeded;
}

void FloatMatrix::print( const char* prefix, const char* suffix )
{
	if( prefix != nullptr )
	{
		printf( "%s\n", prefix );
	}

	int M = numRows();
	int N = numCols();

	for( int i = 0; i < M; ++i )
	{
		for( int j = 0; j < N; ++j )
		{
			printf( "%.4f    ", ( *this )( i, j ) );
		}
		printf( "\n" );
	}
	printf( "\n" );

	if( suffix != nullptr )
	{
		printf( "%s\n", suffix );
	}
}

QString FloatMatrix::toString()
{
	int M = numRows();
	int N = numCols();

	QString out;

	for( int i = 0; i < M; ++i )
	{
		for( int j = 0; j < N; ++j )
		{
			float val = ( *this )( i, j );
			out.append( QString( "%1" ).arg( val, 10, 'g', 4 ) );
		}
		out.append( "\n" );
	}
	return out;
}

//////////////////////////////////////////////////////////////////////////
// Private
//////////////////////////////////////////////////////////////////////////

float FloatMatrix::norm( char whichNorm ) const
{
	int m = numRows();
	int n = numCols();

	return slange
	(
		&whichNorm,
		&m, &n,
		data(),
		&m,
		nullptr
	);
}

FloatMatrix operator + ( const FloatMatrix& a, const FloatMatrix& b )
{
	FloatMatrix c;
	FloatMatrix::add( a, b, c );
	return c;
}

FloatMatrix operator - ( const FloatMatrix& a, const FloatMatrix& b )
{
	FloatMatrix c;
	FloatMatrix::subtract( a, b, c );
	return c;
}

FloatMatrix operator - ( const FloatMatrix& a )
{
	// TODO: multiply( -1, a )
	FloatMatrix c( a );
	for( int k = 0; k < c.numElements(); ++k )
	{
		c[k] = -c[k];
	}
	return c;
}

FloatMatrix operator * ( const FloatMatrix& a, const FloatMatrix& b )
{
	FloatMatrix c;
	FloatMatrix::multiply( a, b, c );
	return c;
}
