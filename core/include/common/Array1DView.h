#pragma once

#include <cstdint>

#include "common/WrapConstPointerT.h"

// A 1D array view that wraps around a raw pointer and does not take ownership.
template< typename T >
class Array1DView
{
public:

    // The null Array1DView:
	// pointer = nullptr, length = 0.
	Array1DView();

	// Create an Array1DView with the default element stride of sizeof( T ).
	Array1DView( void* pPointer, int size );

	// Create an Array1DView with the specified size and element stride.
	Array1DView( void* pPointer, int size, int stride );
	
	bool isNull() const;
	bool notNull() const;

	operator const T* () const;
	operator T* ();

	const T* pointer() const;
	T* pointer();

	T* elementPointer( int x );

	T& operator [] ( int k );

    // The logical size of the array view
	// (i.e., how many elements of type T there are).
    // For a 1D view, width, size, and numElements are all equivalent.
    int width() const;
	int size() const;
    int numElements() const;

	// The space between the start of elements, in bytes.
    // For a 1D view, stride and elementStrideBytes are equivalent.
    int elementStrideBytes() const;
	int stride() const;
	
	// Returns true if the array is tightly packed,
	// i.e. elementStrideBytes() == sizeof( T ).
	bool elementsArePacked() const;
	bool packed() const;

    // Conversion operator to Array1DView< const T >
    operator Array1DView< const T >() const;

private:

	int m_size;
	int m_stride;
	typename WrapConstPointerT< T, uint8_t >::pointer m_pPointer;
};

#include "Array1DView.inl"