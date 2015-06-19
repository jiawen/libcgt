#pragma once

#include <vector>

#include "SparseMatrixCommon.h"

template< typename T >
class CoordinateSparseMatrix;

class FloatMatrix;

// TODO: debug this by serializing to disk in coordinate format

// Compressed Sparse Column (CSC) sparse matrix
template< typename T >
class CompressedSparseMatrix
{
public:	

	CompressedSparseMatrix( MatrixType matrixType = GENERAL, uint32_t nRows = 0, uint32_t nCols = 0, uint32_t nnz = 0 );

	void reset( uint32_t nRows, uint32_t nCols, uint32_t nnz );

	uint32_t numNonZeros() const;
	uint32_t numRows() const;
	uint32_t numCols() const;

	// only valid where the structure is already defined
	// get and put takes O( log( nnz ) ) since it requires a binary search tree lookup
	T get( uint32_t i, uint32_t j ) const;
	void put( uint32_t i, uint32_t j, const T& value );

	MatrixType matrixType() const;

	// the non-zero values of this matrix
	std::vector< T >& values();
	const std::vector< T >& values() const;
	
	// the vector of inner indices
	//   innerIndices has length numNonZeroes(), the same as values().size()
	//   innerIndices[k] is the row index of the k-th non-zero element in values()
	std::vector< uint32_t >& innerIndices();
	const std::vector< uint32_t >& innerIndices() const;

	// the vector of outer index pointers
	// compressed sparse column format:
	//   outerIndexPointers has length numCols() + 1
	//   outerIndexPointers[j] is the index of the first element of column j in values()
	//   outerIndexPointers[j+1] - outerIndices[j] = # non-zero elements in column j
	//   outerIndexPointers[numCols()] = numNonZeroes()
	std::vector< uint32_t >& outerIndexPointers();
	const std::vector< uint32_t >& outerIndexPointers() const;
	
	// returns a data structure mapping indices (i,j)
	// to indices in values() and innerIndices()
	SparseMatrixStructureTreeMap& structureMap();

	void transposed( CompressedSparseMatrix< T >& f ) const;

	// sparse-dense vector product
	// y <-- Ax
	void multiplyVector( FloatMatrix& x, FloatMatrix& y );

	// sparse-dense vector product
	// y <-- A' x
	// A is m x n, so A' is n x m
	// x should be m x 1, y should be n x 1
	void multiplyTransposeVector( FloatMatrix& x, FloatMatrix& y );

	// sparse-sparse product
	// computes A^T A
	// Since the product is always symmetric,
	// only the lower triangle of the output is ever stored
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
	//    size n x n where n = numRows()
	//    matrix type SYMMETRIC
	//
	// TODO: if output format is compressed sparse col, store the lower triangle
	// TODO: if output format is full, do the copy
	void multiplyTranspose( CompressedSparseMatrix< T >& product ) const;	

	// copy data from the same matrix in coordinate format given index map
	void gather( const CoordinateSparseMatrix< T >& coord, const std::vector< int >& indexMap );

	// TODO: multiply sparse * dense using: mkl_?cscmm

	// product <- a * b
	// if product.matrixType() == GENERAL, then the full sparse matrix is stored
	// else, then the lower triangle is stored
	// (set the product.matrixType() to SYMMETRIC only if you know it's going to be symmetric)
	// TODO: implement general --> symmetric that correctly drops?
	static void multiply( const CompressedSparseMatrix< T >& a, const CompressedSparseMatrix< T >& b,
		CompressedSparseMatrix< T >& product );

private:

	MatrixType m_matrixType;

	uint32_t m_nRows;
	uint32_t m_nCols;

	std::vector< T > m_values;
	std::vector< uint32_t > m_innerIndices;
	std::vector< uint32_t > m_outerIndexPointers;

	// structure dictionary
	// mapping matrix coordinates (i,j) --> index k
	// in m_values and m_innerIndices
	SparseMatrixStructureTreeMap m_structureMap;
};
