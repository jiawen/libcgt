#pragma once

#include <cstdio>
#include <cstdint>
#include <cstring>

#include "common/Array1DView.h"

// TODO: strides should be size_t, so always positive

// A simple 1D array class.
template< typename T >
class Array1D
{
public:

	// Default null array with dimensions 0 and no data allocated.
	Array1D();

    // Takes ownership of pointer and views it as a 1D array with stride.
    // All sizes and strides must be positive.
    Array1D( void* pointer, int size );
    Array1D( void* pointer, int size, int stride );

    // Read file from disk.
    Array1D( const char* filename );

    // All sizes and strides must be positive.
    Array1D( int size, const T& fillValue = T() );
    Array1D( int size, int stride, const T& fillValue = T() );

    Array1D( const Array1D< T >& copy );
    Array1D( Array1D< T >&& move );
    Array1D& operator = ( const Array1D< T >& copy );
	Array1D& operator = ( Array1D< T >&& move );
    virtual ~Array1D();	

	bool isNull() const;
	bool notNull() const;
    
    // Makes this array null *without* freeing the underlying memory: it is returned instead.
    // Dimensions are set to 0.
    Array1DView< T > relinquish();

    // Makes this array null and frees the underlying memory.
    // Dimensions are set to 0.
	void invalidate();

    // The logical number of elements in this array.
    int width() const;
	int size() const;
    int numElements() const;

    // The space between the start of elements in bytes.
    int elementStrideBytes() const;
    int stride() const;

	void fill( const T& fillValue );

	// Resizes the array, freeing the original data.
	// If width or height <= 0, the array is invalidated
	void resize( int size );
    void resize( int size, int elementStrideBytes );

    // Get a pointer to the first element.
    const T* pointer() const;
    T* pointer();

    const T* elementPointer( int x ) const;
    T* elementPointer( int x );

    operator Array1DView< const T >() const;
    operator Array1DView< T >();

    operator const T* () const;
	operator T* ();

	const T& operator [] ( int k ) const; // read
	T& operator [] ( int k ); // write

    // only works if T doesn't have pointers, with sizeof() well defined
	bool load( const char* filename );
    bool load( FILE* fp );

	// only works if T doesn't have pointers, with sizeof() well defined
	bool save( const char* filename ) const;
    bool save( FILE* fp ) const;

private:
	
    int m_size;
    int m_stride;
	uint8_t* m_array;
};

#include "Array1D.inl"
