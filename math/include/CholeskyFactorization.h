#pragma once

#include <memory>
#include "FloatMatrix.h"
#include "MatrixCommon.h"

// Cholesky factorization for an n x n symmetric positive definite matrix
// A = L * L' or U' * U
// (L is lower triangular, and U is upper triangular)
class CholeskyFactorization
{
public:

    CholeskyFactorization( const FloatMatrix& a, MatrixTriangle storedTriangle = LOWER );

    // returns the inverse of A
    // (returns the 0x0 matrix if A is singular)
    FloatMatrix inverse() const;
    bool inverse( FloatMatrix& output ) const;

    // once A is factored
    // solve for multiple right hand sides
    FloatMatrix solve( const FloatMatrix& rhs ) const;
    bool solve( const FloatMatrix& rhs, FloatMatrix& solution ) const;

    MatrixTriangle storedTriangle() const;

    bool isValid() const;

    // returns either l or u, depending on storedTriangle()
    const FloatMatrix& factor() const;

private:

    MatrixTriangle m_storedTriangle;
    bool m_valid;

    FloatMatrix m_factor;
};
