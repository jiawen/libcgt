#pragma once

#include <vector>

#include <vecmath/Vector2i.h>
#include <vecmath/Vector3f.h>

#include "MatrixCommon.h"

class QString;

class FloatMatrix
{
public:

    static FloatMatrix zeroes( int m, int n );
    static FloatMatrix ones( int m, int n );

    FloatMatrix(); // makes a 0x0 matrix
    FloatMatrix( int nRows, int nCols, float fillValue = 0.f );

    // initialize a matrix from a row major array of values
    FloatMatrix( int nRows, int nCols, float rowMajorValues[] );
    FloatMatrix( const FloatMatrix& copy );
    FloatMatrix( FloatMatrix&& move ); // move constructor
    virtual ~FloatMatrix();
    FloatMatrix& operator = ( const FloatMatrix& copy );
    FloatMatrix& operator = ( FloatMatrix&& move ); // move assignment operator

    bool isNull() const; // a matrix is null if either nRows or nCols is 0

    // TODO: add an invalidate() that makes it -1 x -1 (or 0x0)

    void fill( float d );

    int numRows() const;
    int numCols() const;
    int numElements() const;
    Vector2i indexToSubscript( int idx ) const;
    int subscriptToIndex( int i, int j ) const;

    // resizes this matrix to nRows x nCols and fills it with 0
    void resize( int nRows, int nCols );

    // returns false if nRows * nCols != numElements()
    bool reshape( int nRows, int nCols );

    // access at ( i, j )
    float& operator () ( int i, int j );
    const float& operator () ( int i, int j ) const;

    // access element k with column-major ordering
    float& operator [] ( int k );
    const float& operator [] ( int k ) const;

    // assigns m to this
    // this is resized if necessary
    void copy( const FloatMatrix& m );

    // TODO: check sizes
    void copyTriangle( const FloatMatrix& src, MatrixTriangle srcTriangle, MatrixTriangle dstTriangle, int k = 0 );

    // extracts a triangular part of this matrix starting with the k-th diagonal
    // k = 0 includes the diagonal
    // k > 0 is above the diagonal
    // k < 0 is below the diagonal
    // (see tril and triu in MATLAB)
    FloatMatrix extractTriangle( MatrixTriangle triangle, int k = 0 ) const;

    // assign a submatrix of m
    // starting at (i0,j0)
    // with size (nRows,nCols)
    //
    // to a submatrix of this
    // starting at (i1,j1)
    //
    // nRows = 0 means i1 to end
    // nCols = 0 means j1 to end
    // TODO: rename to copySubmatrix()
    // extractSubmatrix
    void assign( const FloatMatrix& m, int i0 = 0, int j0 = 0, int i1 = 0, int j1 = 0, int nRows = 0, int nCols = 0 );

    const float* data() const;
    float* data();

    // solves the system Ax = B with LU factorization
    // A = *this, a general square matrix
    // B contains multiple right hand sides
    //
    // returns a 0x0 matrix on failure
    // TODO: return a bool, pass in a matrix
    FloatMatrix solve( const FloatMatrix& rhs ) const;
    FloatMatrix solve( const FloatMatrix& rhs, bool& succeeded ) const;

    // solves the system Ax = B with Cholesky factorization
    // A = *this, a symmetric matrix with full storage
    //    if storedTriangle = LOWER, only the entries in the lower triangle are used
    //    else, the upper triangle
    // B contains multiple right hand sides
    //
    // returns a 0x0 matrix on failure
    // TODO: return a bool, pass in a matrix
    FloatMatrix solveSPD( const FloatMatrix& rhs, MatrixTriangle storedTriangle = LOWER ) const;
    FloatMatrix solveSPD( const FloatMatrix& rhs, bool& succeeded, MatrixTriangle storedTriangle = LOWER ) const;

    // TODO: solveLeastSquare?

    // TODO: this crashes in release mode: somehow the matrix is destroyed before it gets moved
    // and the move constructor copies in a 6 x 1418418343 matrix, oh well
    //FloatMatrix eigenvaluesSymmetric( MatrixTriangle storedTriangle = LOWER ) const;
    //FloatMatrix eigenvaluesSymmetric( bool& succeeded, MatrixTriangle storedTriangle = LOWER ) const;

    bool eigenvaluesSymmetric( FloatMatrix& eigenvalues, MatrixTriangle storedTriangle = LOWER ) const;

    // Returns the inverse of this using LU factorization
    // optional out parameter indicates whether the operation succeeded
    // this matrix must be square
    FloatMatrix inverted( bool* pSucceeded = nullptr ) const;

    // Writes the inverse of this into inv
    // returns whether the operation succeeded
    // this matrix must be square
    bool inverted( FloatMatrix& inv ) const;

    // TODO: in-place transpose, using mkl_simatcopy

    // TODO: out-of-place transpose, using mkl_somatcopy
    void transposed( FloatMatrix& t ) const;
    FloatMatrix transposed() const;

    // Returns the square root of the sum of the square of each element
    float frobeniusNorm() const;

    // Returns the sum of the square of each element
    float frobeniusNormSquared() const;

    // returns the maximum row sum
    float infinityNorm() const;

    // returns the maximum column sum
    float l1Norm() const;

    // Returns the element with largest absolute value
    float maximumNorm() const;

    FloatMatrix& operator += ( const FloatMatrix& x );
    FloatMatrix& operator -= ( const FloatMatrix& x );

    // Returns the dot product of a . b
    // a can be m x 1 or 1 x m
    // b can be m x 1 or 1 x m
    // a and b must be of the same length
    static float dot( const FloatMatrix& a, const FloatMatrix& b );

    // TODO: call scaledAdd?
    static void add( const FloatMatrix& a, const FloatMatrix& b, FloatMatrix& c );
    // TODO: call scaledAdd
    static void subtract( const FloatMatrix& a, const FloatMatrix& b, FloatMatrix& c );

    // y <-- alpha * x + y
    // (x and y should be vectors)
    // y is not resized if it's the wrong size
    static void scaledAdd( float alpha, const FloatMatrix& x, FloatMatrix& y );

    static void multiply( const FloatMatrix& a, const FloatMatrix& b, FloatMatrix& c );

    // c <-- alpha * a * b + beta * c
    // c gets resized if it's the wrong size (and contents lost)
    static void scaledMultiplyAdd( float alpha, const FloatMatrix& a, const FloatMatrix& b, float beta, FloatMatrix& c );

    /*
    void eigenvalueDecomposition( QVector< QVector< float > >* eigen_vector,
        QVector< float >* eigen_value );
    static void homography( QVector< Vector3f > from,
        QVector< Vector3f > to, FloatMatrix& output );
    */

    float minimum() const;
    float maximum() const;

    bool loadTXT( QString filename );
    bool saveTXT( QString filename );

    void print( const char* prefix = nullptr, const char* suffix = nullptr ) const;
    QString toString();

private:

    // calls MKL's ?lange to compute a norm
    // whichNorm can be:
    // 'm' for maximum absolute value
    // 'o' for the 1-norm (maximum column sum), or
    // 'f' for the Frobenius norm (sqrt of sum of squares)
    float norm( char whichNorm ) const;

    int m_nRows;
    int m_nCols;
    std::vector< float > m_data;

};

FloatMatrix operator + ( const FloatMatrix& a, const FloatMatrix& b );
FloatMatrix operator - ( const FloatMatrix& a, const FloatMatrix& b );
FloatMatrix operator - ( const FloatMatrix& a );

FloatMatrix operator * ( const FloatMatrix& a, const FloatMatrix& b );
