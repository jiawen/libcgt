#pragma once

#include "common/BasicTypes.h"
#include "math/Indexing.h"

// a 1D array view that wraps around a raw pointer
// and does not take ownership

template< typename T >
class Array1DView
{
public:

	// create an Array1DView with the default stride of sizeof( T )
	Array1DView( const void* pPointer, int length );
	Array1DView( void* pPointer, int length );

	// create an Array1DView with the specified size and stride
	Array1DView( const void* pPointer, int length, int strideBytes );
	Array1DView( void* pPointer, int length, int strideBytes );
	
	const T* pointer() const;
	T* pointer();

	const T& operator [] ( int k ) const; // read
	T& operator [] ( int k ); // write

	// the logical length of the array view
	// (i.e., how many elements of type T there are)
	int length() const;

	// the space between the start of elements in bytes
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