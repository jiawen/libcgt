#pragma once

#include <map>
#include <vector>

#include "SparseMatrixCommon.h"

template< typename T >
class CoordinateSparseMatrix;

template< typename T >
class CompressedSparseMatrix
{
public:	

	// nOuterIndices = nRows for CSR, nCols for CSC
	CompressedSparseMatrix( MatrixType matrixType = GENERAL, CompressedStorageFormat storageFormat = COMPRESSED_SPARSE_COLUMN,
		uint nRows = 0, uint nCols = 0, uint nnz = 0 );

	void reset( uint nRows, uint nCols, uint nnz );

	uint numNonZeros() const;
	uint numRows() const;
	uint numCols() const;

	// only valid where the structure is already defined	
	T get( uint i, uint j ) const;
	void put( uint i, uint j, const T& value );

	MatrixType matrixType() const;
	CompressedStorageFormat storageFormat() const;

	// the non-zero values of this matrix
	std::vector< T >& values();
	
	// the vector of inner indices
	// CSC:
	//   innerIndices has length numNonZeroes(), the same as values().size()
	//   innerIndices[k] is the row index of the k-th non-zero element in values()
	// 
	// CSR: the same as above, but column indices
	std::vector< uint >& innerIndices();

	// the vector of outer indices
	// CSC:
	//   outerIndices has length numCols() + 1
	//   outerIndices[j] is the index of the first element of column j in values()
	//   outerIndices[j+1] - outerIndices[j] = # non-zero elements in column j
	//   outerIndices[numCols()] = numNonZeroes()
	//
	// CSR: the same as above, but for rows
	std::vector< uint >& outerIndices();

	// returns a data structure mapping indices (i,j)
	// to indices in values() and innerIndices()
	std::map< SparseMatrixKey, uint >& structureMap();

	// sparse-sparse product	
	// let A = this
	// If storageFormat is CSR, computes A A^T
	// If storageFormat is CSC, computes A^T A
	// Since the product is always symmetric,
	// only the lower triangle of the output is ever stored
	//
	// since we store only the lower triangle
	// product.compress() must output to COMPRESSED_SPARSE_COLUMN
	//
	// TODO: allow upper and full storage
	void multiplyTranspose( CoordinateSparseMatrix< T >& product ) const;

	// same as above, but optimized
	// so that if ata has already been compressed once
	// and the multiplication is of the same structure
	// will use the existing structure
	//
	// since we store only the lower triangle
	// product must have:
	//    size n x n where n is:
	//      this->numRows(), if this is COMPRESSED_SPARSE_COLUMN
	//      this->numCols(), if this is COMPRESSED_SPARSE_ROW
	//    matrix type SYMMETRIC
	//    storage format COMPRESSED_SPARSE_COLUMN
	//
	// TODO: if output format is compressed sparse col, store the lower triangle
	// TODO: if output format is full, do the copy
	void multiplyTranspose( CompressedSparseMatrix< T >& product ) const;	

private:

	MatrixType m_matrixType;
	CompressedStorageFormat m_storageFormat;

	uint m_nRows;
	uint m_nCols;

	std::vector< T > m_values;
	std::vector< uint > m_innerIndices;
	std::vector< uint > m_outerIndices;

	// structure dictionary
	// mapping matrix coordinates (i,j) --> index k
	// in m_values and m_innerIndices
	std::map< SparseMatrixKey, uint > m_structureMap;
};
