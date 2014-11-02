template< typename T >
Array2D< T >::Array2D() :

	m_width( 0 ),
	m_height( 0 ),
	m_array( nullptr )

{

}

template< typename T >
Array2D< T >::Array2D( const char* filename ) :

	m_width( 0 ),
	m_height( 0 ),
	m_array( nullptr )

{
	load( filename );
}

template< typename T >
Array2D< T >::Array2D( void* pointer, int width, int height ) :

	m_width( width ),
	m_height( height ),
	m_array( reinterpret_cast< uint8_t* >( pointer ) )

{

}

template< typename T >
Array2D< T >::Array2D( int width, int height, const T& fillValue ) :

	m_width( 0 ),
	m_height( 0 ),
	m_array( nullptr )

{
	resize( width, height );
	fill( fillValue );
}

template< typename T >
Array2D< T >::Array2D( const Vector2i& size, const T& fillValue ) :

	m_width( 0 ),
	m_height( 0 ),
	m_array( nullptr )

{
	resize( size );
	fill( fillValue );
}

template< typename T >
Array2D< T >::Array2D( const Array2D< T >& copy ) :

	m_width( 0 ),
	m_height( 0 ),
	m_array( nullptr )

{
	resize( copy.m_width, copy.m_height );
	if( copy.notNull() )
	{
		memcpy( m_array, copy.m_array, m_width * m_height * sizeof( T ) );
	}
}

template< typename T >
Array2D< T >::Array2D( Array2D< T >&& move )
{
	m_width = move.m_width;
	m_height = move.m_height;
	m_array = move.m_array;

	move.m_width = 0;
	move.m_height = 0;
	move.m_array = nullptr;
}

template< typename T >
Array2D< T >& Array2D< T >::operator = ( const Array2D< T >& copy )
{
	if( this != &copy )
	{
		resize( copy.m_width, copy.m_height );
		if( copy.notNull() )
		{
			memcpy( m_array, copy.m_array, m_width * m_height * sizeof( T ) );
		}
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

		m_width = move.m_width;
		m_height = move.m_height;
		m_array = move.m_array;

		move.m_width = 0;
		move.m_height = 0;
		move.m_array = nullptr;
	}
	return *this;
}

template< typename T >
// virtual
Array2D< T >::~Array2D()
{
	invalidate();
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
void Array2D< T >::invalidate()
{
	m_width = 0;
	m_height = 0;

	if( m_array != nullptr )
	{
		delete[] m_array;
		m_array = nullptr;
	}
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
Vector2i Array2D< T >::size() const
{
	return Vector2i( m_width, m_height );
}

template< typename T >
int Array2D< T >::numElements() const
{
	return m_width * m_height;
}

template< typename T >
size_t Array2D< T >::sizeInBytes() const
{
	return numElements() * sizeof( T );
}

template< typename T >
int Array2D< T >::elementStrideBytes() const
{
	return sizeof( T );
}

template< typename T >
int Array2D< T >::rowStrideBytes() const
{
	return m_width * elementStrideBytes();
}

template< typename T >
void Array2D< T >::fill( const T& fillValue )
{
	int ne = numElements();
	for( int k = 0; k < ne; ++k )
	{
		( *this )[ k ] = fillValue;
	}
}

template< typename T >
void Array2D< T >::resize( int width, int height )
{
	// if we request an invalid size
	// then invalidate this
	if( width <= 0 || height <= 0 )
	{
		invalidate();
	}
	// otherwise, it's a valid size
	else
	{
		// check if the total number of elements is the same
		// if it is, don't reallocate
		if( m_width * m_height != width * height )
		{
			if( m_array != nullptr )
			{
				delete[] m_array;
			}

			m_array = new uint8_t[ width * height * sizeof( T ) ];
		}

		// if the number of elements is the same, the dimensions may be different
		m_width = width;
		m_height = height;
	}
}

template< typename T >
void Array2D< T >::resize( const Vector2i& size )
{
	resize( size.x, size.y );
}

template< typename T >
Array2D< T >::operator const Array2DView< T >() const
{
	return Array2DView< T >( m_array, width(), height(), elementStrideBytes(), rowStrideBytes() );
}

template< typename T >
Array2D< T >::operator Array2DView< T >()
{
	return Array2DView< T >( m_array, width(), height(), elementStrideBytes(), rowStrideBytes() );
}

template< typename T >
Array2D< T >::operator const T* () const
{
	return reinterpret_cast< T* >( m_array );
}

template< typename T >
Array2D< T >::operator T* ()
{
	return reinterpret_cast< T* >( m_array );
}

template< typename T >
const T* Array2D< T >::pointer() const
{
	return reinterpret_cast< T* >( m_array );
}

template< typename T >
T* Array2D< T >::pointer()
{
	return reinterpret_cast< T* >( m_array );
}

template< typename T >
const T* Array2D< T >::elementPointer( int x, int y ) const
{
	return reinterpret_cast< T* >( m_array + y * rowStrideBytes() + x * elementStrideBytes() );
}

template< typename T >
T* Array2D< T >::elementPointer( int x, int y )
{
	return reinterpret_cast< T* >( m_array + y * rowStrideBytes() + x * elementStrideBytes() );
}

template< typename T >
const T* Array2D< T >::rowPointer( int y ) const
{
	return elementPointer( 0, y );
}

template< typename T >
T* Array2D< T >::rowPointer( int y )
{
	return elementPointer( 0, y );
}

template< typename T >
const T& Array2D< T >::operator [] ( int k ) const
{
	int x;
	int y;
	Indexing::indexToSubscript2D( k, m_width, x, y );
	return ( *this )( x, y );
}

template< typename T >
T& Array2D< T >::operator [] ( int k )
{
	int x;
	int y;
	Indexing::indexToSubscript2D( k, m_width, x, y );
	return ( *this )( x, y );
}

template< typename T >
const T& Array2D< T >::operator () ( int x, int y ) const
{
	return *elementPointer( x, y );
}

template< typename T >
T& Array2D< T >::operator () ( int x, int y )
{
	return *elementPointer( x, y );
}

template< typename T >
const T& Array2D< T >::operator [] ( const Vector2i& xy ) const
{
	return ( *this )( xy.x, xy.y );
}

template< typename T >
T& Array2D< T >::operator [] ( const Vector2i& xy )
{
	return ( *this )( xy.x, xy.y );
}

#if 0
template< typename T >
template< typename S >
Array2D< S > Array2D< T >::reinterpretAs( int outputWidth, int outputHeight, int outputrowStrideBytes )
{
	Array2D< S > output;

	// if source is null, then return a null array
	if( isNull() )
	{
		return output;
	}

	// if the requested output widths are not default
	// and the sizes don't fit
	if( outputWidth != 0 || outputHeight != 0 || outputrowStrideBytes != 0 )
	{
		int srcBytes = rowStrideBytes * m_height;
		int dstBytes = outputrowStrideBytes * outputHeight;
		if( srcBytes != dstBytes )
		{
			return output;
		}
	}

	output.m_array = reinterpret_cast< S* >( m_array );

	if( outputWidth == 0 && outputHeight == 0 && outputrowStrideBytes == 0 )
	{
		output.m_width = m_width * sizeof( T ) / sizeof( S );
		output.m_height = m_height;
	}
	else
	{
		output.m_width = outputWidth;
		output.m_height = outputHeight;
	}

	m_array = nullptr;
	m_width = 0;
	m_height = 0;

	return output;
}
#endif

template< typename T >
bool Array2D< T >::load( const char* filename )
{
	FILE* fp = fopen( filename, "rb" );
	if( fp == nullptr )
	{
		return false;
	}

	int whp[3];
	size_t elementsRead;

	elementsRead = fread( whp, sizeof( int ), 3, fp );
	if( elementsRead != 3 )
	{
		return false;
	}

	int width = whp[0];
	int height = whp[1];

	size_t nBytes = width * height;
	uint8_t* pBuffer = new uint8_t[ nBytes ];

	// read elements
	elementsRead = fread( pBuffer, 1, nBytes, fp );
	if( elementsRead != nBytes )
	{
		delete[] pBuffer;
		return false;
	}

	// close file
	int fcloseRetVal = fclose( fp );
	if( fcloseRetVal != 0 )
	{
		delete[] pBuffer;
		return false;
	}

	// read succeeded, swap contents
	m_width = width;
	m_height = height;

	if( m_array != nullptr )
	{
		delete[] m_array;
	}
	m_array = reinterpret_cast< T* >( pBuffer );

	return true;
}

template< typename T >
bool Array2D< T >::save( const char* filename ) const
{
	FILE* fp = fopen( filename, "wb" );
	if( fp == nullptr )
	{
		return false;
	}

	fwrite( &m_width, sizeof( int ), 1, fp );
	fwrite( &m_height, sizeof( int ), 1, fp );
	fwrite( m_array, 1, m_width * m_height, fp );
	fclose( fp );

	return true;
}
