#include "CoordinateSparseMatrix.h"

template< typename T >
CoordinateSparseMatrix< T >::CoordinateSparseMatrix() :

m_nRows( 0 ),
	m_nCols( 0 )

{

}

template< typename T >
CoordinateSparseMatrix< T >::CoordinateSparseMatrix( uint initialCapacity ) :

m_nRows( 0 ),
	m_nCols( 0 )

{
	m_ijv.reserve( initialCapacity );
}

template< typename T >
uint CoordinateSparseMatrix< T >::numNonZeroes() const
{
	return static_cast< uint >( m_ijv.size() );
}

template< typename T >
uint CoordinateSparseMatrix< T >::numRows() const
{
	return m_nRows;
}

template< typename T >
uint CoordinateSparseMatrix< T >::numCols() const
{
	return m_nCols;
}

template< typename T >
void CoordinateSparseMatrix< T >::append( uint i, uint j, const T& value )
{
	CoordinateSparseMatrix< T >::Triplet t;
	t.i = i;
	t.j = j;
	t.value = value;

	m_ijv.push_back( t );

	if( i >= m_nRows )
	{
		m_nRows = i + 1;
	}
	if( j >= m_nCols )
	{
		m_nCols = j + 1;
	}
}

template< typename T >
void CoordinateSparseMatrix< T >::compress( CompressedSparseMatrix< T >& output ) const
{
	std::vector< Triplet > ijvSorted( m_ijv );

	uint m = numRows();
	uint n = numCols();
	uint nnz = numNonZeroes();

	uint innerIndex = 0;
	uint outerIndexIndex = 0;

	// TODO: could just transpose i <--> j and use a single sort (or insert into a std::map)
	// innerIndex == index in sorted list

	if( output.storageFormat() == COMPRESSED_SPARSE_COLUMN )
	{
		std::sort( ijvSorted.begin(), ijvSorted.end(), colMajorLess );

		output.reset( m, n, nnz );

		for( uint k = 0; k < nnz; ++k )
		{
			const Triplet& t = ijvSorted[k];
			uint i = t.i;
			uint j = t.j;
			const T& value = t.value;

			// TODO: if we want the output to be only symmetric or triangular
			//if( ( !upperTriangleOnly ) ||
			// ( j >= i ) )
			{
				output.m_values[ innerIndex ] = value;
				output.m_innerIndices[ innerIndex ] = i;

				output.m_structureMap[ std::make_pair( i, j ) ] = innerIndex;

				if( j == outerIndexIndex )
				{
					output.m_outerIndices[ outerIndexIndex ] = innerIndex;
					++outerIndexIndex;
				}

				++innerIndex;
			}		
		}
		output.m_outerIndices[ outerIndexIndex ] = innerIndex;
		++outerIndexIndex;
	}
	else
	{
		std::sort( ijvSorted.begin(), ijvSorted.end(), rowMajorLess );

		output.reset( m, n, nnz );

		for( uint k = 0; k < nnz; ++k )
		{
			const Triplet& t = ijvSorted[k];
			uint i = t.i;
			uint j = t.j;
			const T& value = t.value;

			// TODO: if we want the output to be only symmetric or triangular
			//if( ( !upperTriangleOnly ) ||
			// ( j >= i ) )
			{
				output.m_values[ innerIndex ] = value;
				output.m_innerIndices[ innerIndex ] = j;

				output.m_structureMap[ std::make_pair( i, j ) ] = innerIndex;

				if( i == outerIndexIndex )
				{
					output.m_outerIndices[ outerIndexIndex ] = innerIndex;
					++outerIndexIndex;
				}

				++innerIndex;				
			}		
		}
		output.m_outerIndices[ outerIndexIndex ] = innerIndex;
		++outerIndexIndex;
	}	
}

// static
template< typename T >
bool CoordinateSparseMatrix< T >::rowMajorLess( Triplet& a, Triplet& b )
{
	if( a.i < b.i )
	{
		return true;
	}
	else if( a.i > b.i )
	{
		return false;
	}
	else
	{
		return( a.j < b.j );
	}
}

// static
template< typename T >
bool CoordinateSparseMatrix< T >::colMajorLess( Triplet& a, Triplet& b )
{
	if( a.j < b.j )
	{
		return true;
	}
	else if( a.j > b.j )
	{
		return false;
	}
	else
	{
		return( a.i < b.i );
	}
}

// instantiate

template
CoordinateSparseMatrix< float >;

template
CoordinateSparseMatrix< double >;