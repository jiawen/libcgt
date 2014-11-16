template< typename T >
Array2D< T >::Array2D() :
    m_size( 0 ),
    m_strides( 0 ),
	m_array( nullptr )

{

}

template< typename T >
Array2D< T >::Array2D( void* pointer, const Vector2i& size ) :
    m_size( size ),
    m_strides( { static_cast< int >( sizeof( T ) ), static_cast< int >( m_size.x * sizeof( T ) ) } ),
    m_array( reinterpret_cast< T* >( pointer ) )
{

}

template< typename T >
Array2D< T >::Array2D( void* pointer, const Vector2i& size, const Vector2i& strides ) :

    m_size( size ),
    m_strides( strides ),
    m_array( reinterpret_cast< T* >( pointer ) )

{

}

template< typename T >
Array2D< T >::Array2D( const char* filename ) :

    m_size( 0 ),
    m_strides( 0 ),
    m_array( nullptr )

{
	load( filename );
}

template< typename T >
Array2D< T >::Array2D( const Vector2i& size, const T& fillValue ) :
    m_size( 0 ),
    m_strides( 0 ),
    m_array( nullptr )
{
	resize( size );
	fill( fillValue );
}

template< typename T >
Array2D< T >::Array2D( const Array2D< T >& copy ) :
    m_size( 0 ),
    m_strides( 0 ),
    m_array( nullptr )
{
    resize( copy.m_size );
	if( copy.notNull() )
	{
		memcpy( m_array, copy.m_array, m_strides.y * m_size.y );
	}
}

template< typename T >
Array2D< T >::Array2D( Array2D< T >&& move )
{
    m_size = move.m_size;
    m_strides = move.m_strides;
	m_array = move.m_array;

    move.m_size = { 0, 0 };
    move.m_strides = { 0, 0 };
	move.m_array = nullptr;
}

template< typename T >
Array2D< T >& Array2D< T >::operator = ( const Array2D< T >& copy )
{
	if( this != &copy )
	{
        resize( copy.m_size );
		if( copy.notNull() )
		{
			memcpy( m_array, copy.m_array, m_strides.y * m_size.y );
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

        m_size = move.m_size;
        m_strides = move.m_strides;
        m_array = move.m_array;

        move.m_size = { 0, 0 };
        move.m_strides = { 0, 0 };
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
    m_size = { 0, 0 };
    m_strides = { 0, 0 };

	if( m_array != nullptr )
	{
		delete[] m_array;
		m_array = nullptr;
	}
}

template< typename T >
int Array2D< T >::width() const
{
	return m_size.x;
}

template< typename T >
int Array2D< T >::height() const
{
	return m_size.y;
}

template< typename T >
Vector2i Array2D< T >::size() const
{
    return m_size;
}

template< typename T >
int Array2D< T >::numElements() const
{
    return m_size.x * m_size.y;
}

template< typename T >
int Array2D< T >::elementStrideBytes() const
{
    return m_strides.x;
}

template< typename T >
int Array2D< T >::rowStrideBytes() const
{
    return m_strides.y;
}

template< typename T >
Vector2i Array2D< T >::strides() const
{
	return m_strides;
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
void Array2D< T >::resize( const Vector2i& size )
{
    resize( size, { static_cast< int >( sizeof( T ) ), static_cast< int >( size.x * sizeof( T ) ) } );
}

template< typename T >
void Array2D< T >::resize( const Vector2i& size, const Vector2i& strides )
{
	// if we request an invalid size
	// then invalidate this
    if( size.x <= 0 || size.y <= 0 || strides.x <= 0 || strides.y <= 0 )
	{
		invalidate();
	}
	// otherwise, it's a valid size
	else
	{
		// Check if the total number of memory is the same.
		// If it is, can reuse memory.
        if( size.y * strides.y != m_size.y * m_strides.y )
		{
			if( m_array != nullptr )
			{
				delete[] m_array;
			}

            uint8_t* pBuffer = new uint8_t[ size.y * strides.y ];
			m_array = reinterpret_cast< T* >( pBuffer );
		}

		// if the number of elements is the same, the dimensions may be different
        m_size = size;
        m_strides = strides;
	}
}

template< typename T >
Array2D< T >::operator Array2DView< const T >() const
{
    return Array2DView< const T >( m_array, m_size, m_strides );
}

template< typename T >
Array2D< T >::operator Array2DView< T >()
{
    return Array2DView< T >( m_array, m_size, m_strides );
}

template< typename T >
const T* Array2D< T >::elementPointer( const Vector2i& xy ) const
{
    return reinterpret_cast< const T* >( &( m_array[ Vector2i::dot( xy, m_strides ) ] ) );
}

template< typename T >
T* Array2D< T >::elementPointer( const Vector2i& xy )
{
    return reinterpret_cast< T* >( &( m_array[ Vector2i::dot( xy, m_strides ) ] ) );
}

template< typename T >
Array2D< T >::operator const T* () const
{
    return reinterpret_cast< const T* >( &( m_array[ y * m_strides.y ] ) );
}

template< typename T >
Array2D< T >::operator T* ()
{
    return reinterpret_cast< T* >( &( m_array[ y * m_strides.y ] ) );
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
	Indexing::indexToSubscript2D( k, m_size.x, x, y );
    return ( *this )[ { x, y } ];
}

template< typename T >
T& Array2D< T >::operator [] ( int k )
{
	int x;
	int y;
	Indexing::indexToSubscript2D( k, m_size.x, x, y );
    return ( *this )[ { x, y } ];
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

template< typename T >
bool Array2D< T >::load( const char* filename )
{
	FILE* fp = fopen( filename, "rb" );
	if( fp == nullptr )
	{
		return false;
	}

    int dims[ 4 ];
	size_t elementsRead;

    elementsRead = fread( dims, sizeof( int ), 4, fp );
	if( elementsRead != 4 )
	{
		return false;
	}

    int width = dims[ 0 ];
    int height = dims[ 1 ];
    int elementStrideBytes = dims[ 2 ];
    int rowStrideBytes = dims[ 3 ];

    size_t nBytes = rowStrideBytes * height;
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
    m_size = { width, height };
    m_strides = { elementStrideBytes, rowStrideBytes };

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

	fwrite( &m_size, sizeof( int ), 2, fp );
    fwrite( &m_strides, sizeof( int ), 2, fp );
	fwrite( m_array, 1, m_size.y * m_strides.y, fp );
	fclose( fp );

	return true;
}
