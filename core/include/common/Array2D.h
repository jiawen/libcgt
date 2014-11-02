#pragma once

#include <cstdio>
#include <cstdint>
#include <cstring>

#include "common/Array2DView.h"
#include "math/Indexing.h"
#include "vecmath/Vector2i.h"

// A simple 2D array class (with tightly packed, row-major storage)
// TODO: allow element and row pitches, but only positive ones.
// TODO: Array1D
template< typename T >
class Array2D
{
public:

	// Default null array with dimensions 0x0 and no data allocated
	Array2D();
	Array2D( const char* filename );
	Array2D( void* pointer, int width, int height ); // Take ownership of a pointer.
	Array2D( int width, int height, const T& fillValue = T() );
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

	size_t sizeInBytes() const;
	int elementStrideBytes() const; // the space between the start of elements, in bytes, equal to sizeof( T )
	int rowStrideBytes() const; // number of bytes between successive rows, equal to width * elementStrideBytes()

	void fill( const T& fillValue );

	// resizes the array, original data is not preserved
	// if width or height <= 0, the array is invalidated
	void resize( int width, int height );
	void resize( const Vector2i& size );

	operator const Array2DView< T >() const;
	operator Array2DView< T >();

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

	const T& operator () ( int x, int y ) const; // read
	T& operator () ( int x, int y ); // write

	const T& operator [] ( const Vector2i& xy ) const; // read
	T& operator [] ( const Vector2i& xy ); // write
	
	// only works if T doesn't have pointers, with sizeof() well defined
	bool load( const char* filename );

	// only works if T doesn't have pointers, with sizeof() well defined
	bool save( const char* filename ) const;

private:
	
	int m_width;
	int m_height;
	uint8_t* m_array;
};

#include "Array2D.inl"
