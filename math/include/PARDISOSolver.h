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
	void analyzePattern( int m, int n, int* rowIndex, int* columns, int nNonZeroes );

	void analyzePattern( CompressedSparseMatrix< valueType >& A );

	// factorize: take in the values, which has the same ordering as setup
	// if sparsity structure has changed, call setup again
	void factorize( valueType* values );

	void factorize( CompressedSparseMatrix< valueType >& A );

	// actually solve
	void solve( const valueType* rhs, valueType* solution );

	void solve( const FloatMatrix& rhs, FloatMatrix& solution );

private:

	_MKL_DSS_HANDLE_t m_handle;

};
