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

	// sparse-sparse product	
	// let A = this
	// If storageFormat is CSR, computes A A^T
	// If storageFormat is CSC, computes A^T A
	// Since the product is always symmetric,
	// only the upper triangle of the output is ever stored
	//
	// since we store only the upper triangle
	// ata.compress() must output to COMPRESSED_SPARSE_ROW
	//
	// TODO: allow lower
	void multiplyTranspose( CoordinateSparseMatrix< T >& ata ) const;

	// same as above, but optimized
	// so that if ata has already been compressed once
	// and the multiplication is of the same structure
	// will use the existing structure
	//
	// since we store only the upper triangle
	// ata must have storage format COMPRESSED_SPARSE_ROW
	void multiplyTranspose( CompressedSparseMatrix< T >& ata ) const;

	std::vector< T > m_values;
	std::vector< uint > m_innerIndices;
	std::vector< uint > m_outerIndices;	

	// structure dictionary
	// mapping matrix coordinates (i,j) --> index k
	// in m_values and m_innerIndices
	std::map< SparseMatrixKey, uint > m_structureMap;

private:

	MatrixType m_matrixType;
	CompressedStorageFormat m_storageFormat;

	uint m_nRows;
	uint m_nCols;
};
