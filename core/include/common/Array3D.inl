template< typename T >
Array3D< T >::Array3D() :
    m_size( 0 ),
    m_strides( 0 ),
	m_array( nullptr )
{

}

template< typename T >
Array3D< T >::Array3D( void* pointer, const Vector3i& size ) :
    m_size( size ),
    m_strides( {
        static_cast< int >( sizeof( T ) ),
        static_cast< int >( size.x * sizeof( T ) ),
        static_cast< int >( size.x * size.y * sizeof( T ) ) }),
    m_array( reinterpret_cast< uint8_t* >( pointer ) )
{

}

template< typename T >
Array3D< T >::Array3D( void* pointer, const Vector3i& size, const Vector3i& strides ) :
    m_size( size ),
    m_strides( strides ),
    m_array( reinterpret_cast< uint8_t* >( pointer ) )
{

}

template< typename T >
Array3D< T >::Array3D( const char* filename ) :
    m_size( 0 ),
    m_strides( 0 ),
    m_array( nullptr )
{
	load( filename );
}

template< typename T >
Array3D< T >::Array3D( const Vector3i& size, const T& fillValue ) :
    m_size( 0 ),
    m_strides( 0 ),
    m_array( nullptr )
{
	resize( size );
	fill( fillValue );
}

template< typename T >
Array3D< T >::Array3D( const Vector3i& size, const Vector3i& strides, const T& fillValue ) :
    m_size( 0 ),
    m_strides( 0 ),
    m_array( nullptr )
{
    resize( size, strides );
    fill( fillValue );
}

template< typename T >
Array3D< T >::Array3D( const Array3D< T >& copy ) :
    m_size( 0 ),
    m_strides( 0 ),
    m_array( nullptr )
{
    resize( copy.m_size, copy.m_strides );
    if( copy.notNull( ) )
    {
        memcpy( m_array, copy.m_array, m_strides.z * m_size.z );
    }
}

template< typename T >
Array3D< T >::Array3D( Array3D< T >&& move )
{
    m_size = move.m_size;
    m_strides = move.m_strides;
    m_array = move.m_array;

    move.m_size = { 0, 0, 0 };
    move.m_strides = { 0, 0, 0 };
    move.m_array = nullptr;
}

template< typename T >
Array3D< T >& Array3D< T >::operator = ( const Array3D< T >& copy )
{
	if( this != &copy )
	{
		resize( copy.m_width, copy.m_height, copy.m_depth );
		if( copy.notNull() )
		{
            memcpy( m_array, copy.m_array, m_strides.z * m_size.z );
		}
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
            delete[ ] m_array;
        }

        m_size = move.m_size;
        m_strides = move.m_strides;
        m_array = move.m_array;

        move.m_size = { 0, 0, 0 };
        move.m_strides = { 0, 0, 0 };
        move.m_array = nullptr;
    }
    return *this;
}

// virtual
template< typename T >
Array3D< T >::~Array3D()
{
	invalidate();
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
Array3DView< T > Array3D< T >::relinquish()
{
    Array3DView< T > output = *this;

    m_size = { 0, 0, 0 };
    m_strides = { 0, 0, 0 };

    return output;
}

template< typename T >
void Array3D< T >::invalidate()
{
    m_size = { 0, 0, 0 };
    m_strides = { 0, 0, 0 };

    if( m_array != nullptr )
    {
        delete[] m_array;
        m_array = nullptr;
    }
}

template< typename T >
int Array3D< T >::width() const
{
	return m_size.x;
}

template< typename T >
int Array3D< T >::height() const
{
	return m_size.y;
}

template< typename T >
int Array3D< T >::depth() const
{
	return m_size.z;
}

template< typename T >
Vector3i Array3D< T >::size() const
{
    return m_size;
}

template< typename T >
int Array3D< T >::numElements() const
{
    return m_size.x * m_size.y * m_size.z;
}

template< typename T >
int Array3D< T >::elementStrideBytes() const
{
    return m_strides.x;
}

template< typename T >
int Array3D< T >::rowStrideBytes() const
{
    return m_strides.y;
}

template< typename T >
int Array3D< T >::sliceStrideBytes() const
{
    return m_strides.z;
}

template< typename T >
Vector3i Array3D< T >::strides() const
{
    return m_strides;
}

template< typename T >
void Array3D< T >::fill( const T& fillValue )
{
	int ne = numElements();
	for( int k = 0; k < ne; ++k )
	{
		( *this )[ k ] = fillValue;
	}
}

template< typename T >
void Array3D< T >::resize( const Vector3i& size )
{
    resize( size, {
        static_cast< int >( sizeof( T ) ),
        static_cast< int >( size.x * sizeof( T ) ),
        static_cast< int >( size.x * size.y * sizeof( T ) )
    } );
}

template< typename T >
void Array3D< T >::resize( const Vector3i& size, const Vector3i& strides )
{
    // Of we request an invalid size then invalidate this.
    if( size.x <= 0 || size.y <= 0 || size.z <= 0 ||
        strides.x <= 0 || strides.y <= 0 || strides.z <= 0 )
    {
        invalidate();
    }
    // Otherwise, it's a valid size.
    else
    {
        // Check if the total number of memory is the same.
        // If it is, can reuse memory.
        if( size.z * strides.z != m_size.z * m_strides.z )
        {
            if( m_array != nullptr )
            {
                delete[] m_array;
            }

            m_array = new uint8_t[ size.z * strides.z ];
        }

        // if the number of elements is the same, the dimensions may be different
        m_size = size;
        m_strides = strides;
    }
}

template< typename T >
const T* Array3D< T >::pointer( ) const
{
    return reinterpret_cast< T* >( m_array );
}

template< typename T >
T* Array3D< T >::pointer( )
{
    return reinterpret_cast< T* >( m_array );
}

template< typename T >
const T* Array3D< T >::elementPointer( const Vector3i& xyz ) const
{
    return reinterpret_cast< const T* >( &( m_array[ Vector3i::dot( xyz, m_strides ) ] ) );
}

template< typename T >
T* Array3D< T >::elementPointer( const Vector3i& xyz )
{
    return reinterpret_cast< T* >( &( m_array[ Vector3i::dot( xyz, m_strides ) ] ) );
}

template< typename T >
const T* Array3D< T >::rowPointer( int y, int z ) const
{
    return elementPointer( { 0, y, z } );
}

template< typename T >
T* Array3D< T >::rowPointer( int y, int z )
{
    return elementPointer( { 0, y, z } );
}

template< typename T >
const T* Array3D< T >::slicePointer( int z ) const
{
    return elementPointer( { 0, 0, z } );
}

template< typename T >
T* Array3D< T >::slicePointer( int z )
{
    return elementPointer( { 0, 0, z } );
}

template< typename T >
Array3D< T >::operator const Array3DView< const T >() const
{
    return Array3DView< const T >( m_array, m_size, m_strides );
}

template< typename T >
Array3D< T >::operator Array3DView< T >()
{
    return Array3DView< T >( m_array, m_size, m_strides );
}

template< typename T >
Array3D< T >::operator const T* () const
{
    return reinterpret_cast< T* >( m_array );
}

template< typename T >
Array3D< T >::operator T* ()
{
    return reinterpret_cast< T* >( m_array );
}

template< typename T >
const T& Array3D< T >::operator [] ( int k ) const
{
	int x;
	int y;
	int z;
	Indexing::indexToSubscript3D( k, m_width, m_height, x, y, z );
    return ( *this )[ { x, y, z } ];
}

template< typename T >
T& Array3D< T >::operator [] ( int k )
{
	int x;
	int y;
	int z;
	Indexing::indexToSubscript3D( k, m_width, m_height, x, y, z );
    return ( *this )[ { x, y, z } ];
}

template< typename T >
const T& Array3D< T >::operator [] ( const Vector3i& xyz ) const
{
    return *( elementPointer( xyz ) );
}

template< typename T >
T& Array3D< T >::operator [] ( const Vector3i& xyz )
{
    return *( elementPointer( xyz ) );
}

template< typename T >
bool Array3D< T >::load( const char* filename )
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
bool Array3D< T >::load( FILE* fp )
{	
	int dims[6];
	size_t elementsRead;
	
    elementsRead = fread( dims, sizeof( int ), 6, fp );
	if( elementsRead != 6 )
	{
		return false;
	}

    int width = dims[ 0 ];
    int height = dims[ 1 ];
    int depth = dims[ 2 ];
    int elementStrideBytes = dims[ 3 ];
    int rowStrideBytes = dims[ 4 ];
    int sliceStrideBytes = dims[ 5 ];

    size_t nBytes = sliceStrideBytes * depth;
	uint8_t* pBuffer = new uint8_t[ nBytes ];

	// read elements
	elementsRead = fread( pBuffer, 1, nBytes, fp );
	if( elementsRead != nBytes )
	{
		delete[] pBuffer;
		return false;
	}

	// read succeeded, swap contents
    m_size = { width, height, depth };
    m_strides = { elementStrideBytes, rowStrideBytes, sliceStrideBytes };
	
	if( m_array != nullptr )
	{
		delete[] m_array;
	}
    m_array = pBuffer;

	return true;
}

template< typename T >
bool Array3D< T >::save( const char* filename )
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
bool Array3D< T >::save( FILE* fp )
{
	// TODO: error checking

	fwrite( &m_width, sizeof( int ), 1, fp );
	fwrite( &m_height, sizeof( int ), 1, fp );
	fwrite( &m_depth, sizeof( int ), 1, fp );
	fwrite( &m_rowPitchBytes, sizeof( int ), 1, fp );
	fwrite( &m_slicePitchBytes, sizeof( int ), 1, fp );
	fwrite( m_array, 1, m_slicePitchBytes * m_depth, fp );

	return true;
}