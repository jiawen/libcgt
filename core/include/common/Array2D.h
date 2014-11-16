#pragma once

#include <cstdio>
#include <cstdint>
#include <cstring>

#include "common/Array2DView.h"
#include "math/Indexing.h"
#include "vecmath/Vector2i.h"

// TODO:
// stride should be settable when taking over something?

// A simple 2D array class (with row-major storage)
template< typename T >
class Array2D
{
public:

	// Default null array with dimensions 0 and no data allocated.
	Array2D();

    // Takes ownership of pointer and views it as a 2D array with strides.
    // All sizes and strides must be positive.
    Array2D( void* pointer, const Vector2i& size );
    Array2D( void* pointer, const Vector2i& size, const Vector2i& strides );

    // Read file from disk.
	Array2D( const char* filename );
	Array2D( const Vector2i& size, const T& fillValue = T() );
	Array2D( const Array2D< T >& copy );
	Array2D( Array2D< T >&& move );
	Array2D& operator = ( const Array2D< T >& copy );
	Array2D& operator = ( Array2D< T >&& move );
    virtual ~Array2D();	

	bool isNull() const;
	bool notNull() const;
	void invalidate(); // makes this array null by setting its dimensions to -1 and frees the data

	int width() const;
	int height() const;
	Vector2i size() const;
	int numElements() const;

    // the space between the start of elements in bytes
    int elementStrideBytes() const;

    // the space between the start of rows in bytes
    int rowStrideBytes() const;

    // { elementStride, rowStride } in bytes.
    Vector2i strides() const;

	void fill( const T& fillValue );

	// Resizes the array, original data is not preserved
	// if width or height <= 0, the array is invalidated
	void resize( const Vector2i& size );
    void resize( const Vector2i& size, const Vector2i& strides );

	operator Array2DView< const T >() const;
	operator Array2DView< T >();

    const T* elementPointer( const Vector2i& xy ) const;
    T* elementPointer( const Vector2i& xy );

	const T* rowPointer( int y ) const;
	T* rowPointer( int y );
	
    operator const T* () const;
	operator T* ();

	const T* pointer() const;
	T* pointer();

	const T* elementPointer( int x, int y ) const;
	T* elementPointer( int x, int y );

	const T* rowPointer( int y ) const;
	T* rowPointer( int y );

	const T& operator [] ( int k ) const; // read
	T& operator [] ( int k ); // write

	const T& operator [] ( const Vector2i& xy ) const; // read
	T& operator [] ( const Vector2i& xy ); // write
	
    // only works if T doesn't have pointers, with sizeof() well defined
	bool load( const char* filename );

	// only works if T doesn't have pointers, with sizeof() well defined
	bool save( const char* filename ) const;

private:
	
    Vector2i m_size;
    Vector2i m_strides;
	T* m_array;
};

#include "Array2D.inl"
