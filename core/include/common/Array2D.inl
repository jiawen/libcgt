template< typename T >
Array2D< T >::Array2D() :

	m_width( -1 ),
	m_height( -1 ),
	m_rowPitchBytes( -1 ),
	m_array( nullptr )

{

}

template< typename T >
Array2D< T >::Array2D( const char* filename ) :

	m_width( -1 ),
	m_height( -1 ),
	m_rowPitchBytes( -1 ),
	m_array( nullptr )

{
	load( filename );
}

template< typename T >
Array2D< T >::Array2D( int width, int height, const T& fillValue ) :

	m_width( -1 ),
	m_height( -1 ),
	m_rowPitchBytes( -1 ),
	m_array( nullptr )

{
	resize( width, height );
	fill( fillValue );
}

template< typename T >
Array2D< T >::Array2D( const Array2D< T >& copy ) :

	m_width( -1 ),
	m_height( -1 ),
	m_rowPitchBytes( -1 ),
	m_array( nullptr )

{
	resize( copy.m_width, copy.m_height );
	memcpy( m_array, copy.m_array, m_rowPitchBytes * m_height );
}

template< typename T >
Array2D< T >::Array2D( Array2D< T >&& move )
{
	m_width = move.m_width;
	m_height = move.m_height;
	m_rowPitchBytes = move.m_rowPitchBytes;
	m_array = move.m_array;

	move.m_width = -1;
	move.m_height = -1;
	move.m_rowPitchBytes = -1;
	move.m_array = nullptr;
}

template< typename T >
Array2D< T >& Array2D< T >::operator = ( const Array2D< T >& copy )
{
	if( this != &copy )
	{
		resize( copy.m_width, copy.m_height );
		memcpy( m_array, copy.m_array, m_rowPitchBytes * m_height );
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
		m_rowPitchBytes = move.m_rowPitchBytes;
		m_array = move.m_array;

		move.m_width = -1;
		move.m_height = -1;
		move.m_rowPitchBytes = -1;
		move.m_array = nullptr;
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
void Array2D< T >::invalidate()
{
	m_width = -1;
	m_height = -1;
	m_rowPitchBytes = -1;

	if( m_array != nullptr )
	{
		delete[] m_array;
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
int Array2D< T >::rowPitchBytes() const
{
	return m_rowPitchBytes;
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
		int rowPitchBytes = width * sizeof( T ); // TODO: round up / align
		if( rowPitchBytes * height != m_rowPitchBytes * m_height )
		{
			if( m_array != nullptr )
			{
				delete[] m_array;
			}

			ubyte* pBuffer = new ubyte[ rowPitchBytes * height ];
			m_array = reinterpret_cast< T* >( pBuffer );
		}

		// if the number of elements is the same, the dimensions may be different
		m_width = width;
		m_height = height;
		m_rowPitchBytes = rowPitchBytes;
	}
}

template< typename T >
void Array2D< T >::resize( const Vector2i& size )
{
	resize( size.x, size.y );
}

template< typename T >
Array2DView< T > Array2D< T >::croppedView( int x, int y )
{
	return croppedView( x, y, width() - x, height() - y );
}

template< typename T >
Array2DView< T > Array2D< T >::croppedView( int x, int y, int width, int height )
{
	T* cornerPointer = &( rowPointer( y )[ x ] );
	return Array2DView< T >( width, height, rowPitchBytes(), cornerPointer );
}

template< typename T >
Array2D< T >::operator const Array2DView< T >() const
{
	return Array2DView< T >( width(), height(), rowPitchBytes(), m_array );
}

template< typename T >
Array2D< T >::operator Array2DView< T >()
{
	return Array2DView< T >( width(), height(), rowPitchBytes(), m_array );
}

template< typename T >
const T* Array2D< T >::rowPointer( int y ) const
{
	ubyte* pBuffer = reinterpret_cast< ubyte* >( m_array );
	return reinterpret_cast< T* >( &( pBuffer[ y * rowPitchBytes() ] ) );
}

template< typename T >
T* Array2D< T >::rowPointer( int y )
{
	ubyte* pBuffer = reinterpret_cast< ubyte* >( m_array );
	return reinterpret_cast< T* >( &( pBuffer[ y * rowPitchBytes() ] ) );
}

template< typename T >
Array2D< T >::operator const T* () const
{
	return m_array;
}

template< typename T >
Array2D< T >::operator T* ()
{
	return m_array;
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
	return rowPointer( y )[ x ];
}

template< typename T >
T& Array2D< T >::operator () ( int x, int y )
{
	return rowPointer( y )[ x ];
}

template< typename T >
const T& Array2D< T >::operator () ( const Vector2i& xy ) const
{
	return rowPointer( xy.y )[ xy.x ];
}

template< typename T >
T& Array2D< T >::operator () ( const Vector2i& xy )
{
	return rowPointer( xy.y )[ xy.x ];
}

template< typename T >
template< typename S >
Array2D< S > Array2D< T >::reinterpretAs( int outputWidth, int outputHeight, int outputRowPitchBytes )
{
	Array2D< S > output;

	// if source is null, then return a null array
	if( isNull() )
	{
		return output;
	}

	// if the requested output widths are not default
	// and the sizes don't fit
	if( outputWidth != -1 || outputHeight != -1 || outputRowPitchBytes != -1 )
	{
		int srcBytes = m_rowPitchBytes * m_height;
		int dstBytes = outputRowPitchBytes * outputHeight;
		if( srcBytes != dstBytes )
		{
			return output;
		}
	}

	output.m_array = reinterpret_cast< S* >( m_array );

	if( outputWidth == -1 && outputHeight == -1 && outputRowPitchBytes == -1 )
	{
		output.m_width = m_width * sizeof( T ) / sizeof( S );
		output.m_height = m_height;
		output.m_rowPitchBytes = outputRowPitchBytes;
	}
	else
	{
		output.m_width = outputWidth;
		output.m_height = outputHeight;
		output.m_rowPitchBytes = outputRowPitchBytes;
	}

	m_array = nullptr;
	m_width = -1;
	m_height = -1;
	m_rowPitchBytes = -1;

	return output;
}

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
	int rowPitchBytes = whp[2];

	size_t nBytes = rowPitchBytes * height;
	ubyte* pBuffer = new ubyte[ nBytes ];

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
	m_rowPitchBytes = rowPitchBytes;

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
	fwrite( &m_rowPitchBytes, sizeof( int ), 1, fp );
	fwrite( m_array, 1, m_rowPitchBytes * m_height, fp );
	fclose( fp );

	return true;
}
