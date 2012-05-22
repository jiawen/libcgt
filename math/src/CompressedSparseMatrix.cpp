#include "CompressedSparseMatrix.h"

#include <cassert>
#include <mkl_spblas.h>

#include <common/Comparators.h>
#include "CoordinateSparseMatrix.h"
#include "FloatMatrix.h"

#include <time/StopWatch.h>

template< typename T >
CompressedSparseMatrix< T >::CompressedSparseMatrix( MatrixType matrixType,
	uint nRows, uint nCols, uint nnz ) :

	m_matrixType( matrixType )

{
	reset( nRows, nCols, nnz );
}

template< typename T >
void CompressedSparseMatrix< T >::reset( uint nRows, uint nCols, uint nnz )
{
	m_nRows = nRows;
	m_nCols = nCols;

	m_values.resize( nnz );
	m_innerIndices.resize( nnz );
	m_outerIndexPointers.resize( nCols + 1 );
	m_structureMap.clear();
}

template< typename T >
uint CompressedSparseMatrix< T >::numNonZeros() const
{
	return static_cast< uint >( m_values.size() );
}

template< typename T >
uint CompressedSparseMatrix< T >::numRows() const
{
	return m_nRows;
}

template< typename T >
uint CompressedSparseMatrix< T >::numCols() const
{
	return m_nCols;
}

template< typename T >
T CompressedSparseMatrix< T >::get( uint i, uint j ) const
{
	T output( 0 );

	auto itr = m_structureMap.find( std::make_pair( i, j ) );
	if( itr != m_structureMap.end() )
	{
		uint k = itr->second;
		output = m_values[ k ];
	}

	return output;
}

template< typename T >
void CompressedSparseMatrix< T >::put( uint i, uint j, const T& value )
{
	uint k = m_structureMap[ std::make_pair( i, j ) ];
	m_values[ k ] = value;
}

template< typename T >
MatrixType CompressedSparseMatrix< T >::matrixType() const
{
	return m_matrixType;
}

template< typename T >
std::vector< T >& CompressedSparseMatrix< T >::values()
{
	return m_values;
}

template< typename T >
const std::vector< T >& CompressedSparseMatrix< T >::values() const
{
	return m_values;
}

template< typename T >
std::vector< uint >& CompressedSparseMatrix< T >::innerIndices()
{
	return m_innerIndices;
}

template< typename T >
const std::vector< uint >& CompressedSparseMatrix< T >::innerIndices() const
{
	return m_innerIndices;
}

template< typename T >
std::vector< uint >& CompressedSparseMatrix< T >::outerIndexPointers()
{
	return m_outerIndexPointers;
}

template< typename T >
const std::vector< uint >& CompressedSparseMatrix< T >::outerIndexPointers() const
{
	return m_outerIndexPointers;
}

template< typename T >
std::map< SparseMatrixKey, uint >& CompressedSparseMatrix< T >::structureMap()
{
	return m_structureMap;
}

template< typename T >
void CompressedSparseMatrix< T >::transposed( CompressedSparseMatrix< T >& f ) const
{
	// TODO: if symmetric... then use MKL!  It's square!

	int m = numRows();
	int n = numCols();
	int nnz = numNonZeros();

	f.reset( n, m, nnz );

	// allocate work space
	// TODO: statically allocate a fairly large amount of work space?
	std::vector< int > work( m, 0 );

	const auto& aV = values();
	const auto& aII = innerIndices();
	const auto& aOIP = outerIndexPointers();

	auto& fV = f.values();
	auto& fII = f.innerIndices();
	auto& fOIP = f.outerIndexPointers();

	// count the number of entries in each row of A
	for( int j = 0 ; j < n; ++j )
	{
		int kStart = aOIP[ j ] ;
		int kEnd = aOIP[ j + 1 ];
		for( int p = kStart; p < kEnd; ++p )
		{
			++( work[ aII[ p ] ] );
		}
	}

	// scan to compute row pointers
	int p = 0;
	for( int i = 0; i < m; ++i )
	{
		fOIP[ i ] = p;
		p += work[ i ];
	}
	fOIP[ m ] = p; // don't forget the last one

	// reset work array to be a copy of the outer index pointers of the output array
	// (except the last one)
	for( int i = 0 ; i < m; ++i )
	{
		work[ i ] = fOIP[ i ];
	}

	// populate fII and fV:
	// sweep through the columns of A
	for( int j = 0; j < n; ++j )
	{
		// the row indices and values for column j are in aII[ kStart ... kEnd - 1 ]
		int kStart = aOIP[ j ];
		int kEnd = aOIP[ j + 1 ];
		for( int p = kStart; p < kEnd; ++p )
		{
			// (i,j) is an entry in A
			int i = aII[ p ];

			// find the next slot row row i and increment it
			int q = work[ i ];
			++( work[ i ] );

			fII[ q ] = j;
			fV[ q ] = aV[ p ];
		}
	}

	// setup output structure map
	for( auto itr = m_structureMap.begin(); itr != m_structureMap.end(); ++itr )
	{
		auto ij = itr->first;
		auto k = itr->second;
		auto ji = std::make_pair( ij.second, ij.first );
		f.m_structureMap[ ji ] = k;
	}
}

template< typename T >
void CompressedSparseMatrix< T >::multiplyVector( FloatMatrix& x, FloatMatrix& y )
{
	assert( matrixType() != TRIANGULAR );

	int m = numRows();
	int n = numCols();

	assert( x.numRows() == n );
	assert( x.numCols() == 1 );
	assert( y.numRows() == m );
	assert( y.numCols() == 1 );

	if( matrixType() == GENERAL )
	{
		// mkl_cspblas_scsrgemv assumes CSR storage
		// since we use CSC, transpose it

		char transa = 't';

		mkl_cspblas_scsrgemv
		(
			&transa,
			&n,
			reinterpret_cast< float* >( m_values.data() ), // HACK: templateize FloatMatrix
			reinterpret_cast< int* >( m_outerIndexPointers.data() ),
			reinterpret_cast< int* >( m_innerIndices.data() ),
			x.data(),
			y.data()
		);
	}
	else if( matrixType() == SYMMETRIC )
	{
		// mkl_cspblas_scsrsymv assumes CSR storage
		// since we use CSC, transpose it
		printf( "IMPLEMENT ME!\n" );
	}
}

template< typename T >
void CompressedSparseMatrix< T >::multiplyTransposeVector( FloatMatrix& x, FloatMatrix& y )
{
	assert( matrixType() == GENERAL );
		
	int m = numRows();
	int n = numCols();

	assert( x.numRows() == m );
	assert( x.numCols() == 1 );
	assert( y.numRows() == n );
	assert( y.numCols() == 1 );

	if( matrixType() == GENERAL )
	{
		// mkl_cspblas_scsrgemv assumes CSR storage
		// since this is the transposed version, don't transpose
	
		char transa = 'n';
	
		mkl_cspblas_scsrgemv
		(
			&transa,
			&n,
			reinterpret_cast< float* >( m_values.data() ), // HACK: templateize FloatMatrix
			reinterpret_cast< int* >( m_outerIndexPointers.data() ),
			reinterpret_cast< int* >( m_innerIndices.data() ),
			x.data(),
			y.data()
		);
	}
}

template< typename T >
void CompressedSparseMatrix< T >::multiplyTranspose( CoordinateSparseMatrix< T >& product ) const
{
	product.clear();
	uint n = static_cast< uint >( m_outerIndexPointers.size() - 1 );

	// CSC: iterate over rows of A' (columns of A)
	// CSR: iterate over rows of A (columns of A')
	for( uint i = 0; i < n; ++i )
	{
		// CSC: iterate over columns of A
		// CSR: iterate over columns of A'
		for( uint j = 0; j <= i; ++j )
		{
			bool nonZero = false;
			T sum( 0 );

			uint k = m_outerIndexPointers[ i ];
			uint kEnd = m_outerIndexPointers[ i + 1 ];

			uint l = m_outerIndexPointers[ j ];
			uint lEnd = m_outerIndexPointers[ j + 1 ];

			while( k < kEnd && l < lEnd )
			{
				uint leftJ = m_innerIndices[ k ];
				uint rightI = m_innerIndices[ l ];

				if( leftJ == rightI )
				{
					T leftValue = m_values[ k ];
					T rightValue = m_values[ l ];
					sum += leftValue * rightValue;
					++k;
					++l;

					nonZero = true;
				}
				else if( leftJ < rightI )
				{
					++k;
				}
				else
				{
					++l;
				}
			}

			// output sum
			if( nonZero )
			{
				product.append( i, j, sum );
			}
		}
	}
}

template< typename T >
void CompressedSparseMatrix< T >::multiplyTranspose( CompressedSparseMatrix< T >& product ) const
{
	uint n = static_cast< uint >( m_outerIndexPointers.size() - 1 );

	assert( product.numRows() == n );
	assert( product.numCols() == n );
	assert( product.matrixType() == SYMMETRIC );

	// CSC: iterate over rows of A' (columns of A)
	// CSR: iterate over rows of A (columns of A')
	for( uint i = 0; i < n; ++i )
	{
		// CSC: iterate over columns of A
		// CSR: iterate over columns of A'
		for( uint j = 0; j <= i; ++j )
		{
			bool nonZero = false;
			T sum( 0 );

			uint k = m_outerIndexPointers[ i ];
			uint kEnd = m_outerIndexPointers[ i + 1 ];

			uint l = m_outerIndexPointers[ j ];
			uint lEnd = m_outerIndexPointers[ j + 1 ];

			while( k < kEnd && l < lEnd )
			{
				uint leftJ = m_innerIndices[ k ];
				uint rightI = m_innerIndices[ l ];

				if( leftJ == rightI )
				{
					T leftValue = m_values[ k ];
					T rightValue = m_values[ l ];
					sum += leftValue * rightValue;
					++k;
					++l;

					nonZero = true;
				}
				else if( leftJ < rightI )
				{
					++k;
				}
				else
				{
					++l;
				}
			}

			// output sum
			if( nonZero )
			{
				product.put( i, j, sum );
			}
		}
	}
}

// static
template< typename T >
void CompressedSparseMatrix< T >::multiply( const CompressedSparseMatrix< T >& a, const CompressedSparseMatrix< T >& b,
	CompressedSparseMatrix< T >& product )
{
	MatrixType productType = product.matrixType();

	// A is m x n
	// B is n x p
	// product C is m x p
	uint m = a.numRows();
	uint n = a.numCols();
	assert( n == b.numRows() );
	uint p = b.numCols();

	// count how many elements are in C
	uint nnzC = 0;
	const auto& aV = a.values();
	const auto& aII = a.innerIndices();
	const auto& aOIP = a.outerIndexPointers();
	const auto& bV = b.values();
	const auto& bII = b.innerIndices();
	const auto& bOIP = b.outerIndexPointers();

	std::vector< bool > flags( m );

	// iterate over columns of B (or C)
	for( uint j = 0; j < p; ++j )
	{
		// clear flag array
		flags.assign( m, false );

		// for column j, see which rows are occupied
		uint k = bOIP[ j ];
		uint kEnd = bOIP[ j + 1 ];
		while( k < kEnd )
		{
			uint bi = bII[ k ];

			// B[ bi, j ] is a non-zero element
			// which means A[ :, bi ] will contribute to the product
			// look for non-zero elements of A[ :, bi ]

			uint l = aOIP[ bi ];
			uint lEnd = aOIP[ bi + 1 ];
			while( l < lEnd )
			{
				uint ai = aII[ l ];
				
				// A[ ai, bi ] is a non-zero element
				// and will contribute to the product
				//
				// count it if the output matrix is general (not symmetric)
				// if it's symmetric, if it's in the lower triangle
				// and don't double count: check the flag and set it				
				if( ( productType == GENERAL ) ||
					( ai >= j ) &&
					!( flags[ ai ] ) )
				{
					flags[ ai ] = true;
					++nnzC;
				}

				++l;
			}
			
			++k;
		}
	}

	product.reset( m, p, nnzC );
	auto& cV = product.values();
	auto& cII = product.innerIndices();
	auto& cOIP = product.outerIndexPointers();

	// TODO: swap and transpose
	// C = (B'*A')' takes anz + bnz + cnz time
	// and C = (A*B)'' takes 2 * cnz time
	bool swapAndTranspose = ( ( a.numNonZeros() + b.numNonZeros() ) < nnzC );
	(void)swapAndTranspose;

	//printf( "annz = %d, bnnz = %d, cnnz = %d, swapAndTranspose = %d\n", a.numNonZeros(), b.numNonZeros(), nnzC, (int)swapAndTranspose );

	nnzC = 0;
	std::vector< T > work( m, 0 );

	// iterate over columns of B (or C) again
	// this time actually accumulating the product
	for( uint j = 0; j < p; ++j )
	{
		// clear flag array
		flags.assign( m, false );

		// start a new column of C
		cOIP[ j ] = nnzC;
		
		uint k = bOIP[ j ];
		uint kEnd = bOIP[ j + 1 ];
		while( k < kEnd )
		{
			const T& bValue = bV[ k ];
			uint bi = bII[ k ];

			// B[ bi, j ] is a non-zero element
			// which means A[ :, bi ] will contribute to the product
			// look for non-zero elements of A[ :, bi ]

			uint l = aOIP[ bi ];
			uint lEnd = aOIP[ bi + 1 ];
			while( l < lEnd )
			{
				const T& aValue = aV[ l ];
				uint ai = aII[ l ];

				// A[ ai, bi ] is a non-zero element
				// and will contribute to the product
				//
				// count it if the output matrix is general (not symmetric)
				// if it's symmetric, if it's in the lower triangle
				// and don't double count: check the flag and set it
				if( ( productType == GENERAL ) ||
					( ai >= j ) &&
					!( flags[ ai ] ) )
				{
					flags[ ai ] = true;
					cII[ nnzC ] = ai;
					++nnzC;
				}
				work[ ai ] += aValue * bValue;

				++l;
			}

			++k;
		}

		// iterate over C[:,j] and gather from work array
		// The inner indices of C[:,j] starts at:
		//    cOIP[j] (which we set at the beginning of this column)
		// and ends at:
		//    nnzC (which we just computed)
		for( uint kk = cOIP[ j ]; kk < nnzC; ++kk )
		{			
			uint ci = cII[ kk ];
			cV[ kk ] = work[ ci ];
			// clear work as we go along
			work[ ci ] = 0;
		}
	}
	// fill out outer index
	cOIP[ p ] = nnzC;

	StopWatch sw;
	std::vector< std::pair< int, T > > work2;
	// TODO: how do I sort 2 parallel arrays using the first?

	// sort columns of output
	// i.e., ensure that the row indices within each column are ascending
	// and that the values match
	for( uint j = 0; j < p; ++j )
	{
		uint k = cOIP[ j ];
		uint kEnd = cOIP[ j + 1 ];
				
		int n = kEnd - k;
		work2.resize( n );
		for( int kk = 0; kk < n; ++kk )
		{
			work2[ kk ] = std::make_pair( cII[ k + kk ], cV[ k + kk ] );
		}
		std::sort( work2.begin(), work2.end(), Comparators::pairFirstElementLess< int, T > );

		for( int kk = 0; kk < n; ++kk )
		{
			cII[ k + kk ] = work2[ kk ].first;
			cV[ k + kk ] = work2[ kk ].second;
		}
	}
	printf( "sorting took %f ms\n", sw.millisecondsElapsed() );
}

// instantiate

template
CompressedSparseMatrix< float >;

template
CompressedSparseMatrix< double >;