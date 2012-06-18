#pragma once

#include <QString>
#include <vector>

#include "vecmath/Vector2i.h"
#include "vecmath/Vector3f.h"

class FloatMatrix
{
public:

	static FloatMatrix zeroes( int m, int n );
	static FloatMatrix ones( int m, int n );

	FloatMatrix(); // makes a 0x0 matrix
	FloatMatrix( int nRows, int nCols, float fillValue = 0.f );
	FloatMatrix( const FloatMatrix& m );
	FloatMatrix( FloatMatrix&& m ); // move constructor
	FloatMatrix( FloatMatrix* m );
	virtual ~FloatMatrix();
	FloatMatrix& operator = ( const FloatMatrix& m );
	FloatMatrix& operator = ( FloatMatrix&& m ); // move assignment operator

	bool isNull() const; // a matrix is null if either nRows or nCols is 0

	void fill( float d );

	int numRows() const;
	int numCols() const;
	int numElements() const;
	Vector2i indexToSubscript( int idx ) const;
	int subscriptToIndex( int i, int j ) const;

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

	// assign a submatrix of m
	// starting at (i0,j0)
	// with size (nRows,nCols)
	//
	// to a submatrix of this
	// starting at (i1,j1)
	//
	// nRows = 0 means i1 to end
	// nCols = 0 means j1 to end
	void assign( const FloatMatrix& m, int i0 = 0, int j0 = 0, int i1 = 0, int j1 = 0, int nRows = 0, int nCols = 0 );

	const float* data() const;
	float* data();

	FloatMatrix solve( const FloatMatrix& rhs, bool* pSucceeded = nullptr ) const;

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

	// Returns the sum of the square of each element
	float frobeniusNorm() const;

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

	void print( const char* prefix = nullptr, const char* suffix = nullptr );
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
