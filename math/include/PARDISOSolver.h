#pragma once

#include <mkl_pardiso.h>

#include <common/BasicTypes.h>

template< typename valueType, bool zeroBased >
class PARDISOSolver
{
public:

	PARDISOSolver();
	virtual ~PARDISOSolver();

	// TODO:	
	// cholmod sparse sparse multiply: correctly store the lower triangle
	// x <--> values
	// i <--> columns
	// p <--> rowIndex
	//
	// Eigen's SparseMatrix:
	// Values <--> values
	// InnerIndices <--> columns
	// OuterIndexPtrs <--> rowIndex

	// analyzePattern: take in matrix sparsity structure
	// and perform fill-reducing ordering (symbolic factorization)
	void analyzePattern( int m, int n, int* rowIndex, int* columns, int nNonZeroes );

	// factorize: take in the values, which has the same ordering as setup
	// if sparsity structure has changed, call setup again
	void factorize( valueType* values );

	// actually solve
	void solve( valueType* rhs, valueType* solution );

private:

	_MKL_DSS_HANDLE_t m_handle;

};
