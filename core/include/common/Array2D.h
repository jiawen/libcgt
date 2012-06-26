#pragma once

#include <cstdio>

// TODO: switch to using std::vector as underlying representation

// A simple 2D array class (with row-major storage)
template< typename T >
class Array2D
{
public:	

	Array2D();
	Array2D( const char* filename );
	Array2D( int width, int height, const T& fill = T() );
	Array2D( const Array2D& copy );
	Array2D( Array2D&& move );
	Array2D& operator = ( const Array2D& copy );
	Array2D& operator = ( Array2D&& move );
	virtual ~Array2D();
	
	bool isNull() const;
	bool notNull() const;

	int width() const;
	int height() const;
	int numElements() const;

	void fill( const T& val );
	void resize( int width, int height );

	T* rowPointer( int y ) const;

	operator T* ();
	operator const T* () const;

	const T& operator () ( int k ) const; // read
	T& operator () ( int k ); // write

	const T& operator () ( int x, int y ) const; // read
	T& operator () ( int x, int y ); // write

	bool load( const char* filename );
	bool save( const char* filename );

private:
	
	int m_width;
	int m_height;
	T* m_array;

};

template< typename T >
Array2D< T >::Array2D() :

	m_width( -1 ),
	m_height( -1 ),
	m_array( nullptr )

{
	
}

template< typename T >
Array2D< T >::Array2D( const char* filename ) :

	m_width( -1 ),
	m_height( -1 ),
	m_array( nullptr )

{
	load( filename );
}

template< typename T >
Array2D< T >::Array2D( int width, int height, const T& fill )
{
	m_width = width;
	m_height = height;
	
	int n = width * height;
	m_array = new T[ width * height ];

	for( int i = 0; i < n; ++i )
	{
		m_array[ i ] = fill;
	}
}

template< typename T >
Array2D< T >::Array2D( const Array2D& copy )
{
	m_width = copy.m_width;
	m_height = copy.m_height;

	m_array = new T[ m_width * m_height ];
	memcpy( m_array, copy.m_array, m_width * m_height * sizeof( T ) );
}

template< typename T >
Array2D< T >::Array2D( Array2D&& move )
{
	m_array = move.m_array;
	m_width = move.m_width;
	m_height = move.m_height;

	move.m_array = nullptr;
	move.m_width = -1;
	move.m_height = -1;
}

template< typename T >
Array2D< T >& Array2D< T >::operator = ( const Array2D< T >& copy )
{
	if( this != &copy )
	{
		if( m_array != nullptr )
		{
			delete[] m_array;
		}
		m_width = copy.m_width;
		m_height = copy.m_height;

		m_array = new T[ m_width * m_height ];
		memcpy( m_array, copy.m_array, m_width * m_height * sizeof( T ) );
	}
	return *this;
}

template< typename T >
Array2D< T >& Array2D< T >::operator = ( Array2D< T >&& move )
{
	if( this != &move )
	{
		if( m_array != nullptr )
		{
			delete[] m_array;
		}

		m_array = move.m_array;
		m_width = move.m_width;
		m_height = move.m_height;

		move.m_array = nullptr;
		move.m_width = -1;
		move.m_height = -1;
	}
	return *this;
}

template< typename T >
// virtual
Array2D< T >::~Array2D()
{
	if( m_array != nullptr )
	{
		delete[] m_array;
	}
}

template< typename T >
bool Array2D< T >::isNull() const
{
	return( m_array == nullptr );
}

template< typename T >
bool Array2D< T >::notNull() const
{
	return( m_array != nullptr );
}

template< typename T >
int Array2D< T >::width() const
{
	return m_width;
}

template< typename T >
int Array2D< T >::height() const
{
	return m_height;
}

template< typename T >
int Array2D< T >::numElements() const
{
	return m_width * m_height;
}

template< typename T >
void Array2D< T >::fill( const T& val )
{
	int ne = numElements();
	for( int i = 0; i < ne; ++i )
	{
		m_array[ i ] = val;
	}
}

template< typename T >
void Array2D< T >::resize( int width, int height )
{
	// TODO: check if width or height < 0

	// check if the total number of elements it the same
	// if it is, don't reallocate
	if( width * height != m_width * m_height )
	{
		if( m_array != nullptr )
		{
			delete[] m_array;
		}
		m_array = new T[ width * height ];
	}

	m_width = width;
	m_height = height;
}

template< typename T >
T* Array2D< T >::rowPointer( int y ) const
{
	return &( m_array[ y * m_width ] );
}

template< typename T >
Array2D< T >::operator T* ()
{
	return m_array;
}

template< typename T >
Array2D< T >::operator const T* () const
{
	return m_array;
}

template< typename T >
const T& Array2D< T >::operator () ( int k ) const
{
	return m_array[ k ];
}

template< typename T >
T& Array2D< T >::operator () ( int k )
{
	return m_array[ k ];
}

template< typename T >
const T& Array2D< T >::operator () ( int x, int y ) const
{
	return m_array[ y * m_width + x ];
}

template< typename T >
T& Array2D< T >::operator () ( int x, int y )
{
	return m_array[ y * m_width + x ];
}

template< typename T >
bool Array2D< T >::load( const char* filename )
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
	if( m_array != nullptr )
	{
		delete[] m_array;
	}

	int width;
	int height;

	fread( &width, sizeof( int ), 1, fp );
	fread( &height, sizeof( int ), 1, fp );

	m_width = width;
	m_height = height;
	m_array = new T[ width * height ];

	fread( m_array, sizeof( T ), width * height, fp );	

	fclose( fp );

	return false;
}

template< typename T >
bool Array2D< T >::save( const char* filename )
{
	FILE* fp = fopen( filename, "wb" );
	if( fp == nullptr )
	{
		return false;
	}

	fwrite( &m_width, sizeof( int ), 1, fp );
	fwrite( &m_height, sizeof( int ), 1, fp );
	fwrite( m_array, sizeof( T ), m_width * m_height, fp );
	fclose( fp );

	return true;
}
