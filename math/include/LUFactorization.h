#pragma once

#include <memory>
#include "FloatMatrix.h"

// LU factorization for a general m x n matrix
// A = P * L * U, where P is a permuatation matrix
// L is lower triangular, and U is upper triangular
class LUFactorization
{
public:
	
	LUFactorization( const FloatMatrix& a );

	bool isValid() const;

	// returns the inverse of A
	// (returns the 0x0 matrix if A is singular)
	FloatMatrix inverse() const;
	bool inverse( FloatMatrix& output ) const;

	// once A is factored
	// solve for multiple right hand sides
	// A must be square
	FloatMatrix solve( const FloatMatrix& rhs ) const;
	bool solve( const FloatMatrix& rhs, FloatMatrix& solution ) const;

	// returns the combined lower and upper factors in the matrix Y
	// Y's lower triangle contains the strictly lower triangular L (without unit diagonal)
	// Y's upper triangle contsins U (with diagonal)
	const FloatMatrix& y() const;

	// returns the LAPACK style pivot vector of length min( m, n )
	// ipiv[i] means row i <--> row ipiv[i] (swapped)
	const std::vector< int >& ipiv() const;

	// the lower triangular part L with unit diagonal
	FloatMatrix l() const;

	// the upper triangular part U
	FloatMatrix u() const;

	// returns the MATLAB style permutation matrix P
	// such that P * A = L * U
	FloatMatrix permutationMatrix() const;

	// returns the MATLAB style pivot vector (minus 1, to be zero-based)
	std::vector< int > pivotVector() const;	

private:

	int m_nRowsA;
	int m_nColsA;

	FloatMatrix m_y;
	std::vector< int > m_ipiv;

	bool m_valid;
};
