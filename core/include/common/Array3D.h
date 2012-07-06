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

	bool load( const char* filename );
	bool save( const char* filename );

private:
	
	int m_width;
	int m_height;
	int m_depth;
	T* m_array;

};

template< typename T >
Array3D< T >::Array3D() :

	m_width( -1 ),
	m_height( -1 ),
	m_depth( -1 ),
	m_array( nullptr )

{
	
}

template< typename T >
Array3D< T >::Array3D( const char* filename ) :

	m_width( -1 ),
	m_height( -1 ),
	m_depth( -1 ),
	m_array( nullptr )

{
	load( filename );
}

template< typename T >
Array3D< T >::Array3D( int width, int height, int depth, const T& fill )
{
	m_width = width;
	m_height = height;
	m_depth = depth;

	int n = width * height * depth;
	m_array = new T[ n ];

	for( int i = 0; i < n; ++i )
	{
		m_array[ i ] = fill;
	}
}

template< typename T >
Array3D< T >::Array3D( const Array3D& copy )
{
	m_width = copy.m_width;
	m_height = copy.m_height;
	m_depth = copy.m_depth;

	m_array = new T[ m_width * m_height * m_depth ];
	memcpy( m_array, copy.m_array, m_width * m_height * m_depth * sizeof( T ) );
}

template< typename T >
Array3D< T >::Array3D( Array3D&& move )
{
	m_array = move.m_array;
	m_width = move.m_width;
	m_height = move.m_height;
	m_depth = move.m_depth;

	move.m_array = nullptr;
	move.m_width = -1;
	move.m_height = -1;
	move.m_depth = -1;
}

template< typename T >
Array3D< T >& Array3D< T >::operator = ( const Array3D< T >& copy )
{
	if( this != &copy )
	{
		if( m_array != nullptr )
		{
			delete[] m_array;
		}

		m_width = copy.m_width;
		m_height = copy.m_height;
		m_depth = copy.m_depth;

		m_array = new T[ m_width * m_height * m_depth ];
		memcpy( m_array, copy.m_array, m_width * m_height * m_depth * sizeof( T ) );
	}
	return *this;
}

template< typename T >
Array3D< T >& Array3D< T >::operator = ( Array3D< T >&& move )
{
	if( this != &move )
	{
		if( m_array != nullptr )
		{
			delete[] m_array;
		}

		m_width = move.m_width;
		m_height = move.m_height;
		m_depth = move.m_depth;

		move.m_array = nullptr;
		move.m_width = -1;
		move.m_height = -1;
		move.m_depth = -1;
	}
	return *this;
}

// virtual
template< typename T >
Array3D< T >::~Array3D()
{
	if( m_array != nullptr )
	{
		delete[] m_array;
	}
}

template< typename T >
bool Array3D< T >::isNull() const
{
	return( m_array == nullptr );
}

template< typename T >
bool Array3D< T >::notNull() const
{
	return( m_array != nullptr );
}

template< typename T >
void Array3D< T >::invalidate()
{
	m_width = -1;
	m_height = -1;
	m_depth = -1;

	if( m_array != nullptr )
	{
		delete[] m_array;
	}
}

template< typename T >
int Array3D< T >::width() const
{
	return m_width;
}

template< typename T >
int Array3D< T >::height() const
{
	return m_height;
}

template< typename T >
int Array3D< T >::depth() const
{
	return m_depth;
}

template< typename T >
Vector3i Array3D< T >::size() const
{
	return Vector3i( m_width, m_height, m_depth );
}

template< typename T >
int Array3D< T >::numElements() const
{
	return m_width * m_height * m_depth;
}

template< typename T >
void Array3D< T >::fill( const T& val )
{
	int ne = numElements();
	for( int i = 0; i < ne; ++i )
	{
		m_array[ i ] = val;
	}
}

template< typename T >
void Array3D< T >::resize( int width, int height, int depth )
{
	if( width <= 0 || height <= 0 || depth <= 0 )
	{
		invalidate();
	}
	else
	{
		// check if the total number of elements it the same
		// if it is, don't reallocate
		if( width * height * depth != m_width * m_height * m_depth )
		{
			if( m_array != nullptr )
			{
				delete[] m_array;
			}
			m_array = new T[ width * height * depth ];
		}

		m_width = width;
		m_height = height;
		m_depth = depth;
	}
}

template< typename T >
void Array3D< T >::resize( const Vector3i& size )
{
	resize( size.x, size.y, size.z );
}

template< typename T >
T* Array3D< T >::rowPointer( int y, int z )
{
	return &( m_array[ z * m_width * m_height + y * m_width ] );
}

template< typename T >
const T* Array3D< T >::rowPointer( int y, int z ) const
{
	return &( m_array[ z * m_width * m_height + y * m_width ] );
}

template< typename T >
T* Array3D< T >::slicePointer( int z )
{
	return &( m_array[ z * m_width * m_height ] );
}

template< typename T >
const T* Array3D< T >::slicePointer( int z ) const
{
	return &( m_array[ z * m_width * m_height ] );
}

template< typename T >
Array3D< T >::operator T* () const
{
	return m_array;
}

template< typename T >
const T& Array3D< T >::operator () ( int k ) const
{
	return m_array[ k ];
}

template< typename T >
T& Array3D< T >::operator () ( int k )
{
	return m_array[ k ];
}

template< typename T >
const T& Array3D< T >::operator () ( int x, int y, int z ) const
{
	int k = subscriptToIndex( x, y, z );
	return m_array[ k ];
}

template< typename T >
T& Array3D< T >::operator () ( int x, int y, int z )
{
	int k = subscriptToIndex( x, y, z );
	return m_array[ k ];
}

template< typename T >
inline int Array3D< T >::subscriptToIndex( int x, int y, int z ) const
{
	return z * m_width * m_height + y * m_width + x;
}

template< typename T >
Vector3i Array3D< T >::indextoSubscript( int k ) const
{
	int wh = m_width * m_height;
	int z = k / wh;

	int ky = k - z * wh;
	int y = ky / m_width;

	int x = ky - y * m_width;
	return Vector3i( x, y, z );
}

template< typename T >
bool Array3D< T >::load( const char* filename )
{
	FILE* fp = fopen( filename, "rb" );
	if( fp == nullptr )
	{
		return false;
	}

	// TODO: ensure that the read was successful
	// then delete the old array, and swap
	m_width = -1;
	m_height = -1;
	m_depth = -1;
	if( m_array != nullptr )
	{
		delete[] m_array;
	}

	int width;
	int height;
	int depth;

	fread( &width, sizeof( int ), 1, fp );
	fread( &height, sizeof( int ), 1, fp );
	fread( &depth, sizeof( int ), 1, fp );

	m_width = width;
	m_height = height;
	m_depth = depth;
	m_array = new T[ width * height * depth ];

	fread( m_array, sizeof( T ), width * height * depth, fp );	

	fclose( fp );

	return false;
}

template< typename T >
bool Array3D< T >::save( const char* filename )
{
	FILE* fp = fopen( filename, "wb" );
	if( fp == nullptr )
	{
		return false;
	}

	fwrite( &m_width, sizeof( int ), 1, fp );
	fwrite( &m_height, sizeof( int ), 1, fp );
	fwrite( &m_depth, sizeof( int ), 1, fp );
	fwrite( m_array, sizeof( T ), m_width * m_height * m_depth, fp );
	fclose( fp );

	return true;
}
