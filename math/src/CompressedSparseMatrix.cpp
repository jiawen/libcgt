#include "CompressedSparseMatrix.h"

#include <cassert>
#include <algorithm>

#include <mkl_spblas.h>

#include <common/Comparators.h>
#include <time/StopWatch.h>

#include "CoordinateSparseMatrix.h"
#include "FloatMatrix.h"


template< typename T >
CompressedSparseMatrix< T >::CompressedSparseMatrix( MatrixType matrixType,
	uint32_t nRows, uint32_t nCols, uint32_t nnz ) :

	m_matrixType( matrixType )

{
	reset( nRows, nCols, nnz );
}

template< typename T >
void CompressedSparseMatrix< T >::reset( uint32_t nRows, uint32_t nCols, uint32_t nnz )
{
	m_nRows = nRows;
	m_nCols = nCols;

	m_values.resize( nnz );
	m_innerIndices.resize( nnz );
	m_outerIndexPointers.resize( nCols + 1 );
	m_structureMap.clear();
}

template< typename T >
uint32_t CompressedSparseMatrix< T >::numNonZeros() const
{
	return static_cast< uint32_t >( m_values.size() );
}

template< typename T >
uint32_t CompressedSparseMatrix< T >::numRows() const
{
	return m_nRows;
}

template< typename T >
uint32_t CompressedSparseMatrix< T >::numCols() const
{
	return m_nCols;
}

template< typename T >
T CompressedSparseMatrix< T >::get( uint32_t i, uint32_t j ) const
{
	T output( 0 );

	auto itr = m_structureMap.find( std::make_pair( i, j ) );
	if( itr != m_structureMap.end() )
	{
		uint32_t k = itr->second;
		output = m_values[ k ];
	}

	return output;
}

template< typename T >
void CompressedSparseMatrix< T >::put( uint32_t i, uint32_t j, const T& value )
{
	uint32_t k = m_structureMap[ std::make_pair( i, j ) ];
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
std::vector< uint32_t >& CompressedSparseMatrix< T >::innerIndices()
{
	return m_innerIndices;
}

template< typename T >
const std::vector< uint32_t >& CompressedSparseMatrix< T >::innerIndices() const
{
	return m_innerIndices;
}

template< typename T >
std::vector< uint32_t >& CompressedSparseMatrix< T >::outerIndexPointers()
{
	return m_outerIndexPointers;
}

template< typename T >
const std::vector< uint32_t >& CompressedSparseMatrix< T >::outerIndexPointers() const
{
	return m_outerIndexPointers;
}

template< typename T >
std::map< SparseMatrixKey, uint32_t >& CompressedSparseMatrix< T >::structureMap()
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
	for( int j = 0; j < n; ++j )
	{
		int kStart = aOIP[ j ];
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
	for( int i = 0; i < m; ++i )
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

	y.resize( m, 1 );

	if( matrixType() == GENERAL )
	{
#if 0
		// mkl_cspblas_scsrgemv assumes CSR storage
		// since we use CSC, call it transposed (set transa to 't')

		char transa = 't';

		mkl_cspblas_scsrgemv
		(
			&transa,
			&m,
			reinterpret_cast< float* >( m_values.data() ), // HACK: templatize FloatMatrix
			reinterpret_cast< int* >( m_outerIndexPointers.data() ),
			reinterpret_cast< int* >( m_innerIndices.data() ),
			x.data(),
			y.data()
		);
#endif

		char transa = 'n';
		float alpha = 1;
		float beta = 0;
		char matdescra[6] =
		{
			'G', // general
			'X', // lower/upper (ignored)
			'X', // main diagonal type (ignored)
			'C', // zero-based indexing ('F' for one-based)
			'X', // ignored
			'X', // ignored
		};

		mkl_scscmv
		(
			&transa,
			&m, &n,
			&alpha,
			matdescra,
			reinterpret_cast< float* >( m_values.data() ), // HACK: templatize FloatMatrix
			reinterpret_cast< int* >( m_innerIndices.data() ),
			reinterpret_cast< int* >( m_outerIndexPointers.data() ),
			reinterpret_cast< int* >( &( m_outerIndexPointers[1] ) ),
			x.data(),
			&beta,
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

	y.resize( n, 1 );

	if( matrixType() == GENERAL )
	{
#if 0
		// mkl_cspblas_scsrgemv assumes CSR storage
		// since this is the transposed version, don't transpose
	
		char transa = 'n';
	
		mkl_cspblas_scsrgemv
		(
			&transa,
			&n,
			reinterpret_cast< float* >( m_values.data() ), // HACK: templatize FloatMatrix
			reinterpret_cast< int* >( m_outerIndexPointers.data() ),
			reinterpret_cast< int* >( m_innerIndices.data() ),
			x.data(),
			y.data()
		);
#endif

		char transa = 't';
		float alpha = 1;
		float beta = 0;
		char matdescra[6] =
		{
			'G', // general
			'X', // lower/upper (ignored)
			'X', // main diagonal type (ignored)
			'C', // zero-based indexing ('F' for one-based)
			'X', // ignored
			'X', // ignored
		};

		mkl_scscmv
		(
			&transa,
			&m, &n,
			&alpha,
			matdescra,
			reinterpret_cast< float* >( m_values.data() ), // HACK: templatize FloatMatrix
			reinterpret_cast< int* >( m_innerIndices.data() ),
			reinterpret_cast< int* >( m_outerIndexPointers.data() ),
			reinterpret_cast< int* >( &( m_outerIndexPointers[1] ) ),
			x.data(),
			&beta,
			y.data()
		);
	}
	else if( matrixType() == SYMMETRIC )
	{
		// mkl_cspblas_scsrsymv assumes CSR storage
		// since we use CSC, transpose it
		// careful with the up/lo
		printf( "IMPLEMENT ME!\n" );
	}
}

template< typename T >
void CompressedSparseMatrix< T >::multiplyTranspose( CoordinateSparseMatrix< T >& product ) const
{
	product.clear();
	uint32_t n = static_cast< uint32_t >( m_outerIndexPointers.size() - 1 );

	// CSC: iterate over rows of A' (columns of A)
	// CSR: iterate over rows of A (columns of A')
	for( uint32_t i = 0; i < n; ++i )
	{
		// CSC: iterate over columns of A
		// CSR: iterate over columns of A'
		for( uint32_t j = 0; j <= i; ++j )
		{
			bool nonZero = false;
			T sum( 0 );

			uint32_t k = m_outerIndexPointers[ i ];
			uint32_t kEnd = m_outerIndexPointers[ i + 1 ];

			uint32_t l = m_outerIndexPointers[ j ];
			uint32_t lEnd = m_outerIndexPointers[ j + 1 ];

			while( k < kEnd && l < lEnd )
			{
				uint32_t leftJ = m_innerIndices[ k ];
				uint32_t rightI = m_innerIndices[ l ];

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
void CompressedSparseMatrix< T >::gather( const CoordinateSparseMatrix< T >& coord, const std::vector< int >& indexMap )
{
	uint32_t nnz = numNonZeros();
	for( uint32_t k = 0; k < nnz; ++k )
	{
		int l = indexMap[ k ];
		auto ijk = coord.get( l );
		m_values[ k ] = ijk.value;
	}
}

template< typename T >
void CompressedSparseMatrix< T >::multiplyTranspose( CompressedSparseMatrix< T >& product ) const
{
	uint32_t n = static_cast< uint32_t >( m_outerIndexPointers.size() - 1 );

	assert( product.numRows() == n );
	assert( product.numCols() == n );
	assert( product.matrixType() == SYMMETRIC );

	// CSC: iterate over rows of A' (columns of A)
	// CSR: iterate over rows of A (columns of A')
	for( uint32_t i = 0; i < n; ++i )
	{
		// CSC: iterate over columns of A
		// CSR: iterate over columns of A'
		for( uint32_t j = 0; j <= i; ++j )
		{
			bool nonZero = false;
			T sum( 0 );

			uint32_t k = m_outerIndexPointers[ i ];
			uint32_t kEnd = m_outerIndexPointers[ i + 1 ];

			uint32_t l = m_outerIndexPointers[ j ];
			uint32_t lEnd = m_outerIndexPointers[ j + 1 ];

			while( k < kEnd && l < lEnd )
			{
				uint32_t leftJ = m_innerIndices[ k ];
				uint32_t rightI = m_innerIndices[ l ];

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
	uint32_t m = a.numRows();
	uint32_t n = a.numCols();
	assert( n == b.numRows() );
	uint32_t p = b.numCols();

	// count how many elements are in C
	uint32_t nnzC = 0;
	const auto& aV = a.values();
	const auto& aII = a.innerIndices();
	const auto& aOIP = a.outerIndexPointers();
	const auto& bV = b.values();
	const auto& bII = b.innerIndices();
	const auto& bOIP = b.outerIndexPointers();

	std::vector< bool > flags( m );

	// iterate over columns of B (or C)
	for( uint32_t j = 0; j < p; ++j )
	{
		// clear flag array
		flags.assign( m, false );

		// for column j, see which rows are occupied
		uint32_t k = bOIP[ j ];
		uint32_t kEnd = bOIP[ j + 1 ];
		while( k < kEnd )
		{
			uint32_t bi = bII[ k ];

			// B[ bi, j ] is a non-zero element
			// which means A[ :, bi ] will contribute to the product
			// look for non-zero elements of A[ :, bi ]

			uint32_t l = aOIP[ bi ];
			uint32_t lEnd = aOIP[ bi + 1 ];
			while( l < lEnd )
			{
				uint32_t ai = aII[ l ];
				
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
	// TODO: if it's symmetric, then don't have to transpose in the end

	//printf( "annz = %d, bnnz = %d, cnnz = %d, swapAndTranspose = %d\n", a.numNonZeros(), b.numNonZeros(), nnzC, (int)swapAndTranspose );

	nnzC = 0;
	std::vector< T > work( m, 0 );

	// iterate over columns of B (or C) again
	// this time actually accumulating the product
	for( uint32_t j = 0; j < p; ++j )
	{
		// clear flag array
		flags.assign( m, false );

		// start a new column of C
		cOIP[ j ] = nnzC;
		
		uint32_t k = bOIP[ j ];
		uint32_t kEnd = bOIP[ j + 1 ];
		while( k < kEnd )
		{
			const T& bValue = bV[ k ];
			uint32_t bi = bII[ k ];

			// B[ bi, j ] is a non-zero element
			// which means A[ :, bi ] will contribute to the product
			// look for non-zero elements of A[ :, bi ]

			uint32_t l = aOIP[ bi ];
			uint32_t lEnd = aOIP[ bi + 1 ];
			while( l < lEnd )
			{
				const T& aValue = aV[ l ];
				uint32_t ai = aII[ l ];

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
		for( uint32_t kk = cOIP[ j ]; kk < nnzC; ++kk )
		{			
			uint32_t ci = cII[ kk ];
			cV[ kk ] = work[ ci ];
			// clear work as we go along
			work[ ci ] = 0;
		}
	}
	// fill out outer index
	cOIP[ p ] = nnzC;

	//StopWatch sw;
	std::vector< std::pair< int, T > > work2;
	// TODO: how do I sort 2 parallel arrays using the first?

	// sort columns of output
	// i.e., ensure that the row indices within each column are ascending
	// and that the values match
	for( uint32_t j = 0; j < p; ++j )
	{
		uint32_t k = cOIP[ j ];
		uint32_t kEnd = cOIP[ j + 1 ];
				
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
	//printf( "sorting took %f ms\n", sw.millisecondsElapsed() );
	// TODO: update structure matrix?
}

//////////////////////////////////////////////////////////////////////////
// Instantiate Templates
//////////////////////////////////////////////////////////////////////////

template
CompressedSparseMatrix< float >;

//template
//CompressedSparseMatrix< double >;