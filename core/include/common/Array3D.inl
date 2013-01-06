template< typename T >
Array3D< T >::Array3D() :

	m_width( -1 ),
	m_height( -1 ),
	m_depth( -1 ),
	m_strideBytes( -1 ),
	m_rowPitchBytes( -1 ),
	m_slicePitchBytes( -1 ),
	m_array( nullptr )

{

}

template< typename T >
Array3D< T >::Array3D( const char* filename ) :

	m_width( -1 ),
	m_height( -1 ),
	m_depth( -1 ),
	m_strideBytes( -1 ),
	m_rowPitchBytes( -1 ),
	m_slicePitchBytes( -1 ),
	m_array( nullptr )

{
	load( filename );
}

template< typename T >
Array3D< T >::Array3D( int width, int height, int depth, const T& fillValue ) :

	m_width( -1 ),
	m_height( -1 ),
	m_depth( -1 ),
	m_strideBytes( -1 ),
	m_rowPitchBytes( -1 ),
	m_slicePitchBytes( -1 ),
	m_array( nullptr )

{
	resize( width, height, depth );
	fill( fillValue );
}

template< typename T >
Array3D< T >::Array3D( const Vector3i& size, const T& fillValue ) :

	m_width( -1 ),
	m_height( -1 ),
	m_depth( -1 ),
	m_strideBytes( -1 ),
	m_rowPitchBytes( -1 ),
	m_slicePitchBytes( -1 ),
	m_array( nullptr )

{
	resize( size );
	fill( fillValue );
}

template< typename T >
Array3D< T >::Array3D( const Array3D< T >& copy ) :

	m_width( -1 ),
	m_height( -1 ),
	m_depth( -1 ),
	m_strideBytes( -1 ),
	m_rowPitchBytes( -1 ),
	m_slicePitchBytes( -1 ),
	m_array( nullptr )

{
	resize( copy.m_width, copy.m_height, copy.m_depth );
	if( copy.notNull() )
	{
		memcpy( m_array, copy.m_array, m_slicePitchBytes * m_depth );
	}
}

template< typename T >
Array3D< T >::Array3D( Array3D< T >&& move )
{
	m_width = move.m_width;
	m_height = move.m_height;
	m_depth = move.m_depth;
	m_strideBytes = move.m_strideBytes;
	m_rowPitchBytes = move.m_rowPitchBytes;
	m_slicePitchBytes = move.m_slicePitchBytes;
	m_array = move.m_array;

	move.m_width = -1;
	move.m_height = -1;
	move.m_depth = -1;
	move.m_strideBytes = -1;
	move.m_rowPitchBytes = -1;
	move.m_slicePitchBytes = -1;
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
			memcpy( m_array, copy.m_array, m_slicePitchBytes * m_depth );
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
			delete[] m_array;
		}

		m_width = move.m_width;
		m_height = move.m_height;
		m_depth = move.m_depth;
		m_strideBytes = move.m_strideBytes;
		m_rowPitchBytes = move.m_rowPitchBytes;
		m_slicePitchBytes = move.m_slicePitchBytes;
		m_array = move.m_array;

		move.m_width = -1;
		move.m_height = -1;
		move.m_depth = -1;
		move.m_strideBytes = -1;
		move.m_rowPitchBytes = -1;
		move.m_slicePitchBytes = -1;
		move.m_array = nullptr;
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
	m_rowPitchBytes = -1;
	m_slicePitchBytes = -1;

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
int Array3D< T >::rowPitchBytes() const
{
	return m_rowPitchBytes;
}

template< typename T >
int Array3D< T >::slicePitchBytes() const
{
	return m_slicePitchBytes;
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
void Array3D< T >::resize( int width, int height, int depth )
{
	// if we request an invalid size
	// then invalidate this
	if( width <= 0 || height <= 0 || depth <= 0 )
	{
		invalidate();
	}
	// otherwise, it's a valid size
	else
	{
		// check if the total number of elements is the same
		// if it is, don't reallocate
		int strideBytes = sizeof( T );
		int rowPitchBytes = width * strideBytes; // TODO: round up / align
		int slicePitchBytes = height * rowPitchBytes; // TODO: round up / align

		if( slicePitchBytes * depth != m_slicePitchBytes * m_depth )
		{
			if( m_array != nullptr )
			{
				delete[] m_array;
			}

			ubyte* pBuffer = new ubyte[ slicePitchBytes * depth ];
			m_array = reinterpret_cast< T* >( pBuffer );
		}

		m_width = width;
		m_height = height;
		m_depth = depth;
		m_strideBytes = strideBytes;
		m_rowPitchBytes = rowPitchBytes;
		m_slicePitchBytes = slicePitchBytes;
	}
}

template< typename T >
void Array3D< T >::resize( const Vector3i& size )
{
	resize( size.x, size.y, size.z );
}

template< typename T >
Array3DView< T > Array3D< T >::croppedView( int x, int y, int z )
{
	return croppedView( x, y, z, width() - x, height() - y, depth() - z );
}

template< typename T >
Array3DView< T > Array3D< T >::croppedView( int x, int y, int z, int width, int height, int depth )
{
	T* cornerPointer = &( rowPointer( y, z )[ x ] );
	return Array3DView< T >( width, height, depth, strideBytes(), rowPitchBytes(), slicePitchBytes(), cornerPointer );
}

template< typename T >
Array3D< T >::operator const Array3DView< T >() const
{
	return Array3DView< T >( width(), height(), depth(), strideBytes(), rowPitchBytes(), slicePitchBytes(), m_array );
}

template< typename T >
Array3D< T >::operator Array3DView< T >()
{
	return Array3DView< T >( width(), height(), depth(), strideBytes(), rowPitchBytes(), slicePitchBytes(), m_array );
}

template< typename T >
const T* Array3D< T >::rowPointer( int y, int z ) const
{
	ubyte* pBuffer = reinterpret_cast< ubyte* >( m_array );
	return reinterpret_cast< T* >
	(
		&( pBuffer[ z * slicePitchBytes() + y * rowPitchBytes() ] )
	);
}

template< typename T >
T* Array3D< T >::rowPointer( int y, int z )
{
	ubyte* pBuffer = reinterpret_cast< ubyte* >( m_array );
	return reinterpret_cast< T* >
	(
		&( pBuffer[ z * slicePitchBytes() + y * rowPitchBytes() ] )
	);
}

template< typename T >
const T* Array3D< T >::slicePointer( int z ) const
{
	ubyte* pBuffer = reinterpret_cast< ubyte* >( m_array );
	return reinterpret_cast< T* >
	(
		&( pBuffer[ z * slicePitchBytes() ] )
	);
}

template< typename T >
T* Array3D< T >::slicePointer( int z )
{
	ubyte* pBuffer = reinterpret_cast< ubyte* >( m_array );
	return reinterpret_cast< T* >
	(
		&( pBuffer[ z * slicePitchBytes() ] )
	);
}

template< typename T >
Array3D< T >::operator const T* () const
{
	return m_array;
}

template< typename T >
Array3D< T >::operator T* ()
{
	return m_array;
}

template< typename T >
const T& Array3D< T >::operator [] ( int k ) const
{
	int x;
	int y;
	int z;
	Indexing::indexToSubscript3D( k, m_width, m_height, x, y, z );
	return ( *this )( x, y, z );
}

template< typename T >
T& Array3D< T >::operator [] ( int k )
{
	int x;
	int y;
	int z;
	Indexing::indexToSubscript3D( k, m_width, m_height, x, y, z );
	return ( *this )( x, y, z );
}

template< typename T >
const T& Array3D< T >::operator () ( int x, int y, int z ) const
{
	return rowPointer( y, z )[ x ];
}

template< typename T >
T& Array3D< T >::operator () ( int x, int y, int z )
{
	return rowPointer( y, z )[ x ];
}

template< typename T >
const T& Array3D< T >::operator [] ( const Vector3i& xyz ) const
{
	return rowPointer( xyz.y, xyz.z )[ xyz.x ];
}

template< typename T >
T& Array3D< T >::operator [] ( const Vector3i& xyz )
{
	return rowPointer( xyz.y, xyz.z )[ xyz.x ];
}

template< typename T >
template< typename S >
Array3D< S > Array3D< T >::reinterpretAs( int outputWidth, int outputHeight, int outputDepth,
	int outputRowPitchBytes, int outputSlicePitchBytes )
{
	Array3D< S > output;

	// if source is null, then return a null array
	if( isNull() )
	{
		return output;
	}

	// if the requested output widths are not default
	// and the sizes don't fit
	if( outputWidth != -1 || outputHeight != -1 || outputDepth != -1 ||
		outputRowPitchBytes != -1 || outputSlicePitchBytes != -1 )
	{
		int srcBytes = m_slicePitchBytes * m_depth;
		int dstBytes = outputSlicePitchBytes * outputDepth;
		if( srcBytes != dstBytes )
		{
			return output;
		}
	}

	output.m_array = reinterpret_cast< S* >( m_array );

	if( outputWidth == -1 && outputHeight == -1 && outputDepth == -1 &&
		outputRowPitchBytes == -1 && outputSlicePitchBytes == -1 )
	{
		output.m_width = m_width * sizeof( T ) / sizeof( S );
		output.m_height = m_height;
		output.m_depth = m_depth;
		output.m_rowPitchBytes = outputRowPitchBytes;
		output.m_slicePitchBytes = outputSlicePitchBytes;
	}
	else
	{
		output.m_width = outputWidth;
		output.m_height = outputHeight;
		output.m_depth = outputDepth;
		output.m_rowPitchBytes = outputRowPitchBytes;
		output.m_slicePitchBytes = outputSlicePitchBytes;
	}

	m_array = nullptr;
	m_width = -1;
	m_height = -1;
	m_depth = -1;
	m_rowPitchBytes = -1;
	m_slicePitchBytes = -1;

	return output;
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
	int whdrpsp[5];
	size_t elementsRead;
	
	elementsRead = fread( whdrpsp, sizeof( int ), 5, fp );
	if( elementsRead != 5 )
	{
		return false;
	}

	int width = whdrpsp[0];
	int height = whdrpsp[1];
	int depth = whdrpsp[2];
	int rowPitchBytes = whdrpsp[3];
	int slicePitchBytes = whdrpsp[4];

	size_t nBytes = slicePitchBytes * depth;
	ubyte* pBuffer = new ubyte[ nBytes ];

	// read elements
	elementsRead = fread( pBuffer, 1, nBytes, fp );
	if( elementsRead != nBytes )
	{
		delete[] pBuffer;
		return false;
	}

	// read succeeded, swap contents
	m_width = width;
	m_height = height;
	m_depth = depth;
	m_rowPitchBytes = rowPitchBytes;
	m_slicePitchBytes = slicePitchBytes;
	
	if( m_array != nullptr )
	{
		delete[] m_array;
	}
	m_array = reinterpret_cast< T* >( pBuffer );

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