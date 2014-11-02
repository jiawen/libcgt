template< typename T >
Array2DView< T >::Array2DView() :

	m_width( 0 ),
	m_height( 0 ),
	m_elementStrideBytes( 0 ),
	m_rowStrideBytes( 0 ),
	m_pPointer( nullptr )

{

}

template< typename T >
Array2DView< T >::Array2DView( void* pPointer, int width, int height ) :

	m_width( width ),
	m_height( height ),
	m_elementStrideBytes( sizeof( T ) ),
	m_rowStrideBytes( width * sizeof( T ) ),
	m_pPointer( reinterpret_cast< uint8_t* >( pPointer ) )

{

}

template< typename T >
Array2DView< T >::Array2DView( void* pPointer, const Vector2i& size ) :

	m_width( size.x ),
	m_height( size.y ),
	m_elementStrideBytes( sizeof( T ) ),
	m_rowStrideBytes( size.x * sizeof( T ) ),
	m_pPointer( reinterpret_cast< uint8_t* >( pPointer ) )

{

}

template< typename T >
Array2DView< T >::Array2DView( void* pPointer, int width, int height, int elementStrideBytes, int rowStrideBytes ) :

	m_width( width ),
	m_height( height ),
	m_elementStrideBytes( elementStrideBytes ),
	m_rowStrideBytes( rowStrideBytes ),
	m_pPointer( reinterpret_cast< uint8_t* >( pPointer ) )

{

}

template< typename T >
Array2DView< T >::Array2DView( void* pPointer, const Vector2i& size, int elementStrideBytes, int rowStrideBytes ) :

	m_width( size.x ),
	m_height( size.y ),
	m_elementStrideBytes( elementStrideBytes ),
	m_rowStrideBytes( rowStrideBytes ),
	m_pPointer( reinterpret_cast< uint8_t* >( pPointer ) )

{

}

template< typename T >
bool Array2DView< T >::isNull() const
{
	return( m_pPointer == nullptr );
}

template< typename T >
bool Array2DView< T >::notNull() const
{
	return( m_pPointer != nullptr );
}

template< typename T >
Array2DView< T >::operator const T* () const
{
	return reinterpret_cast< T* >( m_pPointer );
}

template< typename T >
Array2DView< T >::operator T* ()
{
	return reinterpret_cast< T* >( m_pPointer );
}

template< typename T >
const T* Array2DView< T >::pointer() const
{
	return reinterpret_cast< T* >( m_pPointer );
}

template< typename T >
T* Array2DView< T >::pointer()
{
	return reinterpret_cast< T* >( m_pPointer );
}

template< typename T >
const T* Array2DView< T >::elementPointer( int x, int y ) const
{
	return reinterpret_cast< T* >( m_pPointer + y * rowStrideBytes() + x * elementStrideBytes() );
}

template< typename T >
T* Array2DView< T >::elementPointer( int x, int y )
{
	return reinterpret_cast< T* >( m_pPointer + y * rowStrideBytes() + x * elementStrideBytes() );
}

template< typename T >
T* Array2DView< T >::rowPointer( int y )
{
	return elementPointer( 0, y );
}

template< typename T >
const T* Array2DView< T >::rowPointer( int y ) const
{
	return elementPointer( 0, y );
}

template< typename T >
const T& Array2DView< T >::operator [] ( int k ) const
{
	int x;
	int y;
	Indexing::indexToSubscript2D( k, m_width, x, y );
	return ( *this )( x, y );
}

template< typename T >
T& Array2DView< T >::operator [] ( int k )
{
	int x;
	int y;
	Indexing::indexToSubscript2D( k, m_width, x, y );
	return ( *this )( x, y );
}

template< typename T >
const T& Array2DView< T >::operator () ( int x, int y ) const
{
	return *elementPointer( x, y );
}

template< typename T >
T& Array2DView< T >::operator () ( int x, int y )
{
	return *elementPointer( x, y );
}

template< typename T >
const T& Array2DView< T >::operator [] ( const Vector2i& xy ) const
{
	return ( *this )( xy.x, xy.y );
}

template< typename T >
T& Array2DView< T >::operator [] ( const Vector2i& xy )
{
	return ( *this )( xy.x, xy.y );
}

template< typename T >
int Array2DView< T >::width() const
{
	return m_width;
}

template< typename T >
int Array2DView< T >::height() const
{
	return m_height;
}

template< typename T >
Vector2i Array2DView< T >::size() const
{
	return Vector2i( m_width, m_height );
}

template< typename T >
int Array2DView< T >::numElements() const
{
	return width() * height();
}

template< typename T >
size_t Array2DView< T >::bytesReferenced() const
{
	return sizeof( T ) * numElements();
}

template< typename T >
size_t Array2DView< T >::bytesSpanned() const
{
	return std::abs( rowStrideBytes() ) * height();
}

template< typename T >
int Array2DView< T >::elementStrideBytes() const
{
	return m_elementStrideBytes;
}

template< typename T >
int Array2DView< T >::rowStrideBytes() const
{
	return m_rowStrideBytes;
}

template< typename T >
bool Array2DView< T >::elementsArePacked() const
{
	return elementStrideBytes() == sizeof( T );
}

template< typename T >
bool Array2DView< T >::rowsArePacked() const
{
	return rowStrideBytes() == ( width() * elementStrideBytes() );
}

template< typename T >
bool Array2DView< T >::packed() const
{
	return elementsArePacked() && rowsArePacked();
}
