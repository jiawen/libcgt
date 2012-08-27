#pragma once

#include <cstdio>
#include <vecmath/Vector3i.h>

// TODO: switch to using std::vector as underlying representation

// A simple 3D array class (with row-major storage)
template< typename T >
class Array3D
{
public:	

	// Default null array with dimensions -1 and no data allocated
	Array3D();
	Array3D( const char* filename );
	Array3D( int width, int height, int depth, const T& fill = T() );
	Array3D( const Array3D& copy );
	Array3D( Array3D&& move );
	Array3D& operator = ( const Array3D& copy );
	Array3D& operator = ( Array3D&& move );
	virtual ~Array3D();
	
	bool isNull() const;
	bool notNull() const;
	void invalidate();

	int width() const;
	int height() const;
	int depth() const;
	Vector3i size() const;
	int numElements() const;

	void fill( const T& val );

	// resizing with width, height, or depth <= 0 will invalidate this array
	void resize( int width, int height, int depth );
	void resize( const Vector3i& size );

	// Returns a pointer to the beginning of the y-th row of the z-th slice
	T* rowPointer( int y, int z );
	const T* rowPointer( int y, int z ) const;

	// Returns a pointer to the beginning of the z-th slice
	T* slicePointer( int z );
	const T* slicePointer( int z ) const;

	operator T* () const;

	const T& operator () ( int k ) const; // read
	T& operator () ( int k ); // write

	const T& operator () ( int x, int y, int z ) const; // read
	T& operator () ( int x, int y, int z ); // write

	int subscriptToIndex( int x, int y, int z ) const;
	Vector3i indextoSubscript( int k ) const;
	
	// only works if T doesn't have pointers, with sizeof() well defined
	bool load( const char* filename );

	// only works if T doesn't have pointers, with sizeof() well defined
	bool save( const char* filename );

private:
	
	int m_width;
	int m_height;
	int m_depth;
	T* m_array;

};

#include "Array3D.inl"
