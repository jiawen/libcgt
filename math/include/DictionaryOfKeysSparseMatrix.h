#pragma once

#include <map>
#include <QString>

#include "SparseMatrixCommon.h"

template< typename T >
class CompressedSparseMatrix;

template< typename T >
class DictionaryOfKeysSparseMatrix
{
public:	

	uint numRows() const;
	uint numCols() const;
	uint numNonZeroes() const;

	T operator () ( uint i, uint j ) const;
	void put( uint i, uint j, T value );

	// one-based: useful for FORTRAN-style numerical libraries
	// upperTriangleOnly: if the input is already symmetric and positive definite
	//
	// can only do CSR output
	void compress( CompressedSparseMatrix< T >& output,		
		bool oneBased = false, bool upperTriangleOnly = false ) const;

	QString toString() const;

private:

	// dynamically maintained as the maximum inserted value
	uint m_nRows;
	uint m_nCols;

	std::map< SparseMatrixKey, T > m_values;
};

