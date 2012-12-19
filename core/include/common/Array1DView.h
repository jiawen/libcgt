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
	Array1DView( int length, void* pPointer );

	// create an Array1DView with the specified stride between elements in bytes
	Array1DView( int length, int strideBytes, void* pPointer );
	
	const T* pointer() const;
	T* pointer();

	const T& operator [] ( int k ) const; // read
	T& operator [] ( int k ); // write

	int length() const;
	int strideBytes() const;
	
	// returns true if the array is tightly packed
	// i.e. strideBytes == sizeof( T )
	bool packed() const;

private:

	int m_length;
	int m_strideBytes;
	ubyte* m_pPointer;
};

#include "Array1DView.inl"