template< typename T >
Array1D< T >::Array1D() :
    m_size( 0 ),
    m_stride( 0 ),
	m_array( nullptr )
{

}

template< typename T >
Array1D< T >::Array1D( void* pointer, int size ) :
    m_size( size ),
    m_stride( static_cast< int >( sizeof( T ) ) ),
    m_array( reinterpret_cast< uint8_t* >( pointer ) )
{

}

template< typename T >
Array1D< T >::Array1D( void* pointer, int size, int stride ) :
    m_size( size ),
    m_stride( stride ),
    m_array( reinterpret_cast< uint8_t* >( pointer ) )
{

}

template< typename T >
Array1D< T >::Array1D( const char* filename ) :
    m_size( 0 ),
    m_stride( 0 ),
    m_array( nullptr )
{
	load( filename );
}

template< typename T >
Array1D< T >::Array1D( int size, const T& fillValue ) :
    m_size( 0 ),
    m_stride( 0 ),
    m_array( nullptr )
{
    resize( size );
    fill( fillValue );
}

template< typename T >
Array1D< T >::Array1D( int size, int stride, const T& fillValue ) :
    m_size( 0 ),
    m_stride( 0 ),
    m_array( nullptr )
{
	resize( size, stride );
	fill( fillValue );
}

template< typename T >
Array1D< T >::Array1D( const Array1D< T >& copy ) :
    m_size( 0 ),
    m_stride( 0 ),
    m_array( nullptr )
{
    resize( copy.m_size, copy.m_stride );
	if( copy.notNull() )
	{
		memcpy( m_array, copy.m_array, m_stride.y * m_size.y );
	}
}

template< typename T >
Array1D< T >::Array1D( Array1D< T >&& move )
{
    m_size = move.m_size;
    m_stride = move.m_stride;
	m_array = move.m_array;

    move.m_size = 0;
    move.m_stride = 0;
	move.m_array = nullptr;
}

template< typename T >
Array1D< T >& Array1D< T >::operator = ( const Array1D< T >& copy )
{
	if( this != &copy )
	{
        resize( copy.m_size, copy.m_stride );
		if( copy.notNull() )
		{
			memcpy( m_array, copy.m_array, m_stride.y * m_size.y );
		}
	}
	return *this;
}

template< typename T >
Array1D< T >& Array1D< T >::operator = ( Array1D< T >&& move )
{
	if( this != &move )
	{
		if( m_array != nullptr )
		{
			delete[] m_array;
		}

        m_size = move.m_size;
        m_stride = move.m_stride;
        m_array = move.m_array;

        move.m_size = 0;
        move.m_stride = 0;
        move.m_array = nullptr;
	}
	return *this;
}

template< typename T >
// virtual
Array1D< T >::~Array1D()
{
	invalidate();
}

template< typename T >
bool Array1D< T >::isNull() const
{
	return( m_array == nullptr );
}

template< typename T >
bool Array1D< T >::notNull() const
{
	return( m_array != nullptr );
}

template< typename T >
Array1DView< T > Array1D< T >::relinquish()
{
    Array1DView< T > output = *this;

    m_size = 0;
    m_stride = 0;

    return output;
}

template< typename T >
void Array1D< T >::invalidate()
{
    m_size = 0;
    m_stride = 0;

	if( m_array != nullptr )
	{
		delete[] m_array;
		m_array = nullptr;
	}
}

template< typename T >
int Array1D< T >::width() const
{
    return m_size;
}

template< typename T >
int Array1D< T >::size() const
{
    return m_size;
}

template< typename T >
int Array1D< T >::numElements() const
{
    return m_size;
}

template< typename T >
int Array1D< T >::elementStrideBytes() const
{
	return m_stride;
}

template< typename T >
int Array1D< T >::stride() const
{
	return m_stride;
}

template< typename T >
void Array1D< T >::fill( const T& fillValue )
{
	int ne = size();
	for( int k = 0; k < ne; ++k )
	{
		( *this )[ k ] = fillValue;
	}
}

template< typename T >
void Array1D< T >::resize( int size )
{
    resize( size, static_cast< int >( sizeof( T ) ) );
}

template< typename T >
void Array1D< T >::resize( int size, int stride )
{
	// Of we request an invalid size then invalidate this.
    if( size <= 0 || stride <= 0 )
	{
		invalidate();
	}
	// Otherwise, it's a valid size.
	else
	{
		// Check if the total number of memory is the same.
		// If it is, can reuse memory.
        if( size * stride != m_size * m_stride )
		{
			if( m_array != nullptr )
			{
				delete[] m_array;
			}

            m_array = new uint8_t[ size * stride ];
		}

		// if the number of elements is the same, the dimensions may be different
        m_size = size;
        m_stride = stride;
	}
}

template< typename T >
const T* Array1D< T >::pointer() const
{
	return reinterpret_cast< T* >( m_array );
}

template< typename T >
T* Array1D< T >::pointer()
{
	return reinterpret_cast< T* >( m_array );
}

template< typename T >
const T* Array1D< T >::elementPointer( int x ) const
{
    return reinterpret_cast< const T* >( &( m_array[ x * m_stride ] ) );
}

template< typename T >
T* Array1D< T >::elementPointer( int x )
{
    return reinterpret_cast< T* >( &( m_array[ x * m_stride ] ) );
}

template< typename T >
Array1D< T >::operator Array1DView< const T >() const
{
    return Array1DView< const T >( m_array, m_size, m_stride );
}

template< typename T >
Array1D< T >::operator Array1DView< T >()
{
    return Array1DView< T >( m_array, m_size, m_stride );
}

template< typename T >
Array1D< T >::operator const T* () const
{
    return reinterpret_cast< const T* >( m_array );
}

template< typename T >
Array1D< T >::operator T* ()
{
    return reinterpret_cast< T* >( m_array );
}

template< typename T >
const T& Array1D< T >::operator [] ( int k ) const
{
    return *( elementPointer( k ) );
}

template< typename T >
T& Array1D< T >::operator [] ( int k )
{
    return *( elementPointer( k ) );
}

template< typename T >
bool Array1D< T >::load( const char* filename )
{
    FILE* fp = fopen( filename, "rb" );
    if( fp == nullptr )
    {
        return false;
    }

    bool succeeded = load( fp );

    // close file
    int fcloseRetVal = fclose( fp );
    if( fcloseRetVal != 0 )
    {
        return false;
    }

    return succeeded;
}

template< typename T >
bool Array1D< T >::load( FILE* fp )
{
    int dims[ 2 ];
	size_t elementsRead;

    elementsRead = fread( dims, sizeof( int ), 2, fp );
	if( elementsRead != 2 )
	{
		return false;
	}

    int width = dims[ 0 ];
    int elementStrideBytes = dims[ 1 ];

    size_t nBytes = elementStrideBytes * width;
	uint8_t* pBuffer = new uint8_t[ nBytes ];

	// read elements
	elementsRead = fread( pBuffer, 1, nBytes, fp );
	if( elementsRead != nBytes )
	{
		delete[] pBuffer;
		return false;
	}

	// read succeeded, swap contents
    m_size = width;
    m_stride = elementStrideBytes;

	if( m_array != nullptr )
	{
		delete[] m_array;
	}
    m_array = pBuffer;

	return true;
}

template< typename T >
bool Array1D< T >::save( const char* filename ) const
{
    FILE* fp = fopen( filename, "wb" );
    if( fp == nullptr )
    {
        return false;
    }

    bool succeeded = save( fp );
    fclose( fp );
    return succeeded;
}

template< typename T >
bool Array1D< T >::save( FILE* fp ) const
{
    // TODO: error checking

	fwrite( &m_size, sizeof( int ), 1, fp );
    fwrite( &m_stride, sizeof( int ), 1, fp );
	fwrite( m_array, 1, m_size.x * m_stride.x, fp );
	fclose( fp );

	return true;
}
