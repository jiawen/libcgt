#pragma once

#include "common/BasicTypes.h"
#include "math/Indexing.h"

// a 1D array view that wraps around a raw pointer
// and does not take ownership

template< typename T >
class Array1DView
{
public:

	// create an Array1DView with the default stride = sizeof( T )
	Array1DView( int length, T* pPointer );

	// create an Array1DView with the specified stride between elements in bytes
	Array1DView( int length, int strideBytes, T* pPointer );
	
	const T* pointer() const;
	T* pointer();

	const T& operator [] ( int k ) const; // read
	T& operator [] ( int k ); // write
	
private:

	int m_length;
	int m_strideBytes;
	T* m_pPointer;
};

#include "Array2DView.inl"