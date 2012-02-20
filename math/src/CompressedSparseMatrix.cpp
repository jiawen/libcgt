#include "CompressedSparseMatrix.h"

#include <assert.h>
#include "CoordinateSparseMatrix.h"

template< typename T >
CompressedSparseMatrix< T >::CompressedSparseMatrix( MatrixType matrixType, CompressedStorageFormat storageFormat,
	uint nRows, uint nCols, uint nnz ) :

	m_matrixType( matrixType ),
	m_storageFormat( storageFormat )

{
	reset( nRows, nCols, nnz );
}

template< typename T >
void CompressedSparseMatrix< T >::reset( uint nRows, uint nCols, uint nnz )
{
	m_nRows = nRows;
	m_nCols = nCols;

	uint nOuterIndices;
	if( storageFormat() == COMPRESSED_SPARSE_COLUMN )
	{
		nOuterIndices = nCols;
	}
	else
	{
		nOuterIndices = nRows;
	}

	m_values.resize( nnz );
	m_innerIndices.resize( nnz );
	m_outerIndices.resize( nOuterIndices + 1 );
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
CompressedStorageFormat CompressedSparseMatrix< T >::storageFormat() const
{
	return m_storageFormat;
}

template< typename T >
void CompressedSparseMatrix< T >::multiplyTranspose( CoordinateSparseMatrix< T >& ata ) const
{
	uint n = static_cast< uint >( m_outerIndices.size() - 1 );

	// iterate over rows of A'
	for( uint i = 0; i < n; ++i )
	{
		// iterate over columns of A (i.e., the same thing1)
		for( uint j = i; j < n; ++j )
		{
			bool nonZero = false;
			T sum( 0 );

			uint k = m_outerIndices[ i ];
			uint kEnd = m_outerIndices[ i + 1 ];

			uint l = m_outerIndices[ j ];
			uint lEnd = m_outerIndices[ j + 1 ];

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
				ata.append( i, j, sum );
			}
		}
	}
}

template< typename T >
void CompressedSparseMatrix< T >::multiplyTranspose( CompressedSparseMatrix< T >& ata ) const
{
	assert( ata.matrixType() == SYMMETRIC );
	assert( ata.storageFormat() == COMPRESSED_SPARSE_ROW );

	uint n = static_cast< uint >( m_outerIndices.size() - 1 );

	// iterate over rows of A'
	for( uint i = 0; i < n; ++i )
	{
		// iterate over columns of A (i.e., the same thing1)
		for( uint j = i; j < n; ++j )
		{
			bool nonZero = false;
			T sum( 0 );

			uint k = m_outerIndices[ i ];
			uint kEnd = m_outerIndices[ i + 1 ];

			uint l = m_outerIndices[ j ];
			uint lEnd = m_outerIndices[ j + 1 ];

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
				ata.put( i, j, sum );
			}
		}
	}
}

// instantiate

template
CompressedSparseMatrix< float >;

template
CompressedSparseMatrix< double >;