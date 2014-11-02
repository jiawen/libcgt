#pragma once

#include <cstdint>

// a 1D array view that wraps around a raw pointer
// and does not take ownership

template< typename T >
class Array1DView
{
public:

	// pointer = nullptr, length = 0
	Array1DView();

	// create an Array1DView with the default stride of sizeof( T )
	Array1DView( void* pPointer, int length );

	// create an Array1DView with the specified size and stride
	Array1DView( void* pPointer, int length, int elementStrideBytes );
	
	bool isNull() const;
	bool notNull() const;

	operator const T* () const;
	operator T* ();

	const T* pointer() const;
	T* pointer();

	const T* elementPointer( int x ) const;
	T* elementPointer( int x );

	const T& operator [] ( int k ) const; // read
	T& operator [] ( int k ); // write

	// the logical length of the array view
	// (i.e., how many elements of type T there are)
	int length() const;

	// how many bytes does it take if this view were packed.
	// equal to numElements() * sizeof( T )
	size_t bytesReferenced() const;

	// how many bytes does this view span:
	// the total number of bytes in a rectangular region
	// that view overlaps, including the empty spaces.
	// Equal to abs( elementStrideBytes() ) * length()
	size_t bytesSpanned() const;

	// The space between the start of elements, in bytes.
	int elementStrideBytes() const;
	
	// returns true if the array is tightly packed
	// i.e. elementStrideBytes() == sizeof( T )
	bool packed() const;

private:

	int m_length;
	int m_strideBytes;
	uint8_t* m_pPointer;
};

#include "Array1DView.inl"