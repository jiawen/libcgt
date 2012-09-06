#pragma once

#include <cstdio>

#include "common/BasicTypes.h"
#include <math/Indexing.h>
#include <vecmath/Vector2i.h>

// TODO: switch to using std::vector as underlying representation?

// A simple 2D array class (with row-major storage)
template< typename T >
class Array2D
{
public:	

	// Default null array with dimensions -1 and no data allocated
	Array2D();
	Array2D( const char* filename );
	Array2D( int width, int height, const T& fill = T() );
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

	void fill( const T& val );

	// resizing with width or height <= 0 will invalidate this array
	void resize( int width, int height );
	void resize( const Vector2i& size );

	T* rowPointer( int y );
	const T* rowPointer( int y ) const;

	operator T* ();
	operator const T* () const;

	const T& operator () ( int k ) const; // read
	T& operator () ( int k ); // write

	const T& operator () ( int x, int y ) const; // read
	T& operator () ( int x, int y ); // write

	// reinterprets this array as an array of another format,
	// destroying this array
	//
	// by default (outputWidth and outputHeight = -1)
	// the output width is width() * sizeof( T ) / sizeof( S )
	// (i.e., a 3 x 4 x float4 gets cast to a 12 x 4 x float1)
	//
	// If the source is null or the desired output size is invalid
	// returns the null array.
	template< typename S >
	Array2D< S > reinterpretAs( int outputWidth = -1, int outputHeight = -1 );

	// only works if T doesn't have pointers, with sizeof() well defined
	bool load( const char* filename );

	// only works if T doesn't have pointers, with sizeof() well defined
	bool save( const char* filename );

private:
	
	int m_width;
	int m_height;
	T* m_array;

	// to allow reinterpretAs< S >
	template< typename S >
	friend class Array2D;
};

#include "Array2D.inl"