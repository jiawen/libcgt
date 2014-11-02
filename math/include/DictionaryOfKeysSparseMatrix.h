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

	uint32_t numRows() const;
	uint32_t numCols() const;
	uint32_t numNonZeroes() const;

	T operator () ( uint32_t i, uint32_t j ) const;
	void put( uint32_t i, uint32_t j, T value );

	// one-based: useful for FORTRAN-style numerical libraries
	// upperTriangleOnly: if the input is already symmetric and positive definite
	void compress( CompressedSparseMatrix< T >& output,		
		bool oneBased = false, bool upperTriangleOnly = false ) const;

	QString toString() const;

private:

	// dynamically maintained as the maximum inserted value
	uint32_t m_nRows;
	uint32_t m_nCols;

	std::map< SparseMatrixKey, T, SparseMatrixKeyColMajorLess > m_values;
};
