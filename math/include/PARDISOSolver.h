#pragma once

#include <mkl_pardiso.h>

#include <common/BasicTypes.h>

template< typename valueType >
class CompressedSparseMatrix;

class FloatMatrix;

template< typename valueType, bool zeroBased >
class PARDISOSolver
{
public:

	PARDISOSolver();
	virtual ~PARDISOSolver();

	// Eigen's SparseMatrix:
	// Values <--> values
	// InnerIndices <--> columns
	// OuterIndexPtrs <--> rowIndex
	
	// analyzePattern: take in matrix sparsity structure
	// and perform fill-reducing ordering (symbolic factorization)
	bool analyzePattern( int m, int n, int* rowIndex, int* columns, int nNonZeroes );

	bool analyzePattern( CompressedSparseMatrix< valueType >& A );

	// factorize: take in the values, which has the same ordering as setup
	// if sparsity structure has changed, then you need to call analyzePattern again
	bool factorize( valueType* values );

	bool factorize( CompressedSparseMatrix< valueType >& A );

	// actually solve
	bool solve( const valueType* rhs, valueType* solution );

	// solution is automatically resized to A.numRows() x 1
	bool solve( const FloatMatrix& rhs, FloatMatrix& solution );

private:

	int m_nRowsA;
	int m_nColsA;
	_MKL_DSS_HANDLE_t m_handle;

};
