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
void CoordinateSparseMatrix< T >::clear()
{
	m_nRows = 0;
	m_nCols = 0;
	m_ijv.clear();
}

template< typename T >
void CoordinateSparseMatrix< T >::reserve( uint nnz )
{
	m_ijv.reserve( nnz );
}

template< typename T >
void CoordinateSparseMatrix< T >::compress( CompressedSparseMatrix< T >& output ) const
{
	uint m = numRows();
	uint n = numCols();
	uint nnz = numNonZeroes();
	output.reset( m, n, nnz );

	// TODO: if we want the output to be only symmetric or triangular
	//if( ( !upperTriangleOnly ) ||
	// ( j >= i ) )

	// copy the triplet vector
	std::vector< Triplet > ijvSorted( m_ijv );

	if( output.storageFormat() == COMPRESSED_SPARSE_COLUMN )
	{
		for( uint k = 0; k < nnz; ++k )
		{
			Triplet& t = ijvSorted[k];
			std::swap( t.i, t.j );
		}
	}	

	std::sort( ijvSorted.begin(), ijvSorted.end(), rowMajorLess );

	uint innerIndex = 0;
	uint outerIndexIndex = 0;

	auto& values = output.values();
	auto& innerIndices = output.innerIndices();
	auto& outerIndices = output.outerIndices();

	for( uint k = 0; k < nnz; ++k )
	{
		const Triplet& t = ijvSorted[k];
		uint i = t.i;
		uint j = t.j;
		const T& value = t.value;
		
		values[ innerIndex ] = value;
		innerIndices[ innerIndex ] = j;

		if( i == outerIndexIndex )
		{
			outerIndices[ outerIndexIndex ] = innerIndex;
			++outerIndexIndex;
		}
		++innerIndex;				
	}
	outerIndices[ outerIndexIndex ] = innerIndex;
	++outerIndexIndex;

	// populate structure map
	auto& structureMap = output.structureMap();
	if( output.storageFormat() == COMPRESSED_SPARSE_ROW )
	{
		for( uint k = 0; k < nnz; ++k )
		{
			const Triplet& t = ijvSorted[k];
			uint i = t.i;
			uint j = t.j;
			structureMap[ std::make_pair( i, j ) ] = k;
		}
	}
	else
	{
		for( uint k = 0; k < nnz; ++k )
		{
			const Triplet& t = ijvSorted[k];
			uint i = t.j;
			uint j = t.i;
			structureMap[ std::make_pair( i, j ) ] = k;
		}
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