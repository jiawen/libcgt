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
Array2D< T >::Array2D( int width, int height, const T& fill ) :

	m_width( width ),
	m_height( height )

{
	int n = width * height;
	// to make new work without default constructor
	ubyte* pBuffer = new ubyte[ n * sizeof( T ) ];
	m_array = reinterpret_cast< T* >( pBuffer );

	for( int i = 0; i < n; ++i )
	{
		m_array[ i ] = fill;
	}
}

template< typename T >
Array2D< T >::Array2D( const Array2D< T >& copy )
{
	m_width = copy.m_width;
	m_height = copy.m_height;

	m_array = new T[ m_width * m_height ];
	memcpy( m_array, copy.m_array, m_width * m_height * sizeof( T ) );
}

template< typename T >
Array2D< T >::Array2D( Array2D< T >&& move )
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
void Array2D< T >::invalidate()
{
	m_width = -1;
	m_height = -1;

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
	if( width <= 0 || height <= 0 )
	{
		invalidate();
	}
	else
	{
		// check if the total number of elements is the same
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
}

template< typename T >
void Array2D< T >::resize( const Vector2i& size )
{
	resize( size.x, size.y );
}

template< typename T >
T* Array2D< T >::rowPointer( int y )
{
	return &( m_array[ y * m_width ] );
}

template< typename T >
const T* Array2D< T >::rowPointer( int y ) const
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
	return m_array[ Indexing::subscriptToIndex( x, y ) ];
}

template< typename T >
T& Array2D< T >::operator () ( int x, int y )
{
	return m_array[ Indexing::subscriptToIndex( x, y ) ];
}

template< typename T >
template< typename S >
Array2D< S > Array2D< T >::reinterpretAs( int outputWidth, int outputHeight )
{
	Array2D< S > output;

	// if source is null, then return a null array
	if( isNull() )
	{
		return output;
	}

	// if the requested output widths are not default
	// and the sizes don't fit
	if( outputWidth != -1 || outputHeight != -1 )
	{
		int srcBytes = m_width * m_height * sizeof( T );
		int dstBytes = outputWidth * outputHeight * sizeof( S );
		if( srcBytes != dstBytes )
		{
			return output;
		}
	}

	output.m_array = reinterpret_cast< S* >( m_array );

	if( outputWidth == -1 && outputHeight == -1 )
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
	m_width = -1;
	m_height = -1;

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

	int whd[2];
	size_t elementsRead;

	elementsRead = fread( whd, sizeof( int ), 2, fp );
	if( elementsRead != 2 )
	{
		return false;
	}

	int width = whd[0];
	int height = whd[1];

	// to make new work without default constructor
	int nElements = width * height;
	ubyte* pBuffer = new ubyte[ nElements * sizeof( T ) ];
	T* pArray = reinterpret_cast< T* >( pBuffer );

	// read elements
	elementsRead = fread( pArray, sizeof( T ), nElements, fp );
	if( elementsRead != nElements )
	{
		delete[] pArray;
		return false;
	}

	// close file
	int fcloseRetVal = fclose( fp );
	if( fcloseRetVal != 0 )
	{
		delete[] pArray;
		return false;
	}

	// read succeeded, swap contents
	m_width = width;
	m_height = height;

	if( m_array != nullptr )
	{
		delete[] m_array;
	}

	m_array = pArray;

	return true;
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
