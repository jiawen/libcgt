#pragma once

#include <algorithm>

#include "CompressedSparseMatrix.h"

template< typename T >
class CoordinateSparseMatrix
{
public:

	CoordinateSparseMatrix();
	CoordinateSparseMatrix( uint initialCapacity );

	uint numNonZeroes() const;
	uint numRows() const;
	uint numCols() const;

	void append( uint i, uint j, const T& value );
	void clear();

	void compress( CompressedSparseMatrix< T >& output ) const;

private:

	struct Triplet
	{
		uint i;
		uint j;
		T value;
	};

	// compare i first, then j
	static bool rowMajorLess( Triplet& a, Triplet& b );

	// compare j first, then i
	static bool colMajorLess( Triplet& a, Triplet& b );

	// dynamically maintained as the maximum of the appended values
	uint m_nRows;
	uint m_nCols;

	std::vector< Triplet > m_ijv;
};
