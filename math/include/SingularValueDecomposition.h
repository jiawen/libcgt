#pragma once

#include "FloatMatrix.h"

// TODO: return an inverse (requires FloatMatrix::( diag ) )
// TODO: return a pseudoinverse
//
// TODO: economy sized svd?
// TODO: MATLAB's svds
//
// TODO: eigenvalue decomposition using ssyev
// TODO: selected eigenvalues using ssyevx

class SingularValueDecomposition
{
public:

    // Computes the SVD of A:
    // A = U S V^T
    // For A: m x n
    // U will be resized to m x m
    // S will be resized to min( m, n ) x 1
    // Vt will be resized to n x n
    static bool SVD( const FloatMatrix& a, FloatMatrix& u, FloatMatrix& s, FloatMatrix& vt );

    SingularValueDecomposition( const FloatMatrix& a );

    bool isValid() const;

    const FloatMatrix& u() const;
    const FloatMatrix& s() const;
    const FloatMatrix& vt() const;

private:

    FloatMatrix m_u;
    FloatMatrix m_s;
    FloatMatrix m_vt;

    bool m_valid;
};
