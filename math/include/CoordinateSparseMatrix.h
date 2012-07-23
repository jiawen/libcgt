#pragma once

#include <QString>

#include "SparseMatrixCommon.h"
#include "CompressedSparseMatrix.h"

// TODO: dynamically maintain nRows / nCols is expensive?

template< typename T >
class CoordinateSparseMatrix
{
public:

	CoordinateSparseMatrix();
	CoordinateSparseMatrix( int initialCapacity );
	CoordinateSparseMatrix( const CoordinateSparseMatrix& copy );
	CoordinateSparseMatrix& operator = ( const CoordinateSparseMatrix& copy );

	int numNonZeroes() const;
	int numRows() const;
	int numCols() const;

	void append( int i, int j, const T& value );
	void clear();

	// TODO: return a SparseMatrixTriplet< T >
	// gets the k-th appended value
	SparseMatrixTriplet< T > get( int k ) const;

	// reserve memory for at least nnz SparseMatrixTriplet< T >s
	void reserve( int nnz );

	// A <-- A'
	void transpose();

	// f <-- A'
	void transposed( CoordinateSparseMatrix< T >& f ) const;

	// TODO: int removeDuplicates()
	// returns number of duplicates removed?

	void compress( CompressedSparseMatrix< T >& output ) const;
	void compressTranspose( CompressedSparseMatrix< T >& outputAt ) const;	

	// The k-th entry (i,j,value) in this sparse matrix
	// corresponds to the entry output(i,j) at index:
	//   output.values()[ k ] lives at this->m_ijv[ indexMap[ k ] ]
	void compress( CompressedSparseMatrix< T >& output, std::vector< int >& indexMap ) const;

	// The k-th entry (i,j,value) in this sparse matrix
	// corresponds to the entry output(j,i) at index:
	//   output.values()[ k ] lives at this->m_ijv[ indexMap[ k ] ]
	void compressTranspose( CompressedSparseMatrix< T >& outputAt, std::vector< int >& indexMap ) const;

	// sparse-dense vector product
	// y <-- Ax
	void multiplyVector( FloatMatrix& x, FloatMatrix& y );

	// sparse-dense vector product
	// y <-- A'x
	void multiplyTransposeVector( FloatMatrix& x, FloatMatrix& y );

	// TODO: multiplyMatrix with mkl_?coomm

	bool loadTXT( QString filename );
	bool saveTXT( QString filename );

private:

	// TODO: use the three array variation
	// SparseMatrixTriplet< T >s are used only for sorting during compression anyway
	
	void compressCore( std::vector< SparseMatrixTriplet< T > > ijvSorted, CompressedSparseMatrix< T >& output ) const;

	// compare i first, then j
	static bool rowMajorLess( SparseMatrixTriplet< T >& a, SparseMatrixTriplet< T >& b );

	// compare j first, then i
	static bool colMajorLess( SparseMatrixTriplet< T >& a, SparseMatrixTriplet< T >& b );

	// dynamically maintained as the maximum of the appended values
	int m_nRows;
	int m_nCols;

	std::vector< int > m_rowIndices;
	std::vector< int > m_colIndices;
	std::vector< T > m_values;
};
