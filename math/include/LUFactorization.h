#pragma once

#include <memory>
#include "FloatMatrix.h"

// LU factorization for a general m x n matrix
// A = P * L * U, where P is a permuatation matrix
// L is lower triangular, and U is upper triangular
class LUFactorization
{
public:
	
	static std::shared_ptr< LUFactorization > LU( const FloatMatrix& a );

	// output <- inverse of A
	// returns whether inverse succeeded
	bool inverse( FloatMatrix& output );

	// return stuff in matlab style:
	// const FloatMatrix& Y = lu(A), no permutation matrix
	//
	// [l,u] = lu(A): upper triangular matrix u
	// and permuted lower triangula rmatrix l, such that A = l*u, l = lower * permute
	//
	// [l,u,p] = L*U = P*A, u is upper triangular, l is lower triangular with unit diagonal, 
	// l and u should extractable from y, P should be extractable from ipiv
	//
	// ipiv vector of length min( m, n )
	// ipiv[i] means row i <--> row ipiv[i]
	//
	// don't store m_l and m_u
	// return an std::tuple<>

private:

	LUFactorization( const FloatMatrix& a );

	int m_nRowsA;
	int m_nColsA;

	FloatMatrix m_y;
	std::vector< int > m_ipiv;

	FloatMatrix m_l;
	FloatMatrix m_u;
};
