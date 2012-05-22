#pragma once

#include <algorithm>
#include <QString>

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

	// reserve memory for at least nnz triplets
	void reserve( uint nnz );

	// TODO: int removeDuplicates()
	// returns number of duplicates removed?

	void compress( CompressedSparseMatrix< T >& output ) const;
	void compressTranspose( CompressedSparseMatrix< T >& outputAt ) const;

	// The k-th entry (i,j,value) in this sparse matrix
	// corresponds to the entry output(i,j) at index:
	//   l = indexMap[k]
	//   output.values()[ l ]
	//   output.innerIndices()[ l ]
	void compress( CompressedSparseMatrix< T >& output, std::vector< int >& indexMap ) const;

	// The k-th entry (i,j,value) in this sparse matrix
	// corresponds to the entry output(j,i) at index:
	//   l = indexMap[k]
	//   output.values()[ l ]
	//   output.innerIndices()[ l ]
	void compressTranspose( CompressedSparseMatrix< T >& outputAt, std::vector< int >& indexMap ) const;

	// TODO: multiplyVector with mkl_cspblas_?coogemv
	// TODO: multiplyMatrix with mkl_?coomm

	bool saveTXT( QString filename );

private:

	struct Triplet
	{
		uint i;
		uint j;
		T value;
	};

	void compressCore( std::vector< Triplet > ijvSorted, CompressedSparseMatrix< T >& output ) const;

	// compare i first, then j
	static bool rowMajorLess( Triplet& a, Triplet& b );

	// compare j first, then i
	static bool colMajorLess( Triplet& a, Triplet& b );

	// dynamically maintained as the maximum of the appended values
	uint m_nRows;
	uint m_nCols;

	std::vector< Triplet > m_ijv;
};
