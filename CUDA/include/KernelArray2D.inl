template< typename T >
__inline__  __device__
KernelArray2D< T >::KernelArray2D() :

	md_pPitchedPointer( nullptr ),
	m_width( -1 ),
	m_height( -1 ),
	m_pitch( 0 )

{

}

template< typename T >
__inline__  __device__
KernelArray2D< T >::KernelArray2D( T* d_pPitchedPointer, int width, int height, size_t pitch ) :

	md_pPitchedPointer( d_pPitchedPointer ),
	m_width( width ),
	m_height( height ),
	m_pitch( pitch )

{

}

template< typename T >
__inline__  __device__
KernelArray2D< T >::KernelArray2D( T* d_pPitchedPointer, int width, int height ) :

	md_pPitchedPointer( d_pPitchedPointer ),
	m_width( width ),
	m_height( height ),
	m_pitch( width * sizeof( T ) )

{

}

template< typename T >
__inline__  __device__
const T* KernelArray2D< T >::rowPointer( int y ) const
{
	const ubyte* p = reinterpret_cast< const ubyte* >( md_pPitchedPointer );

	// TODO: switch pointer arithmetic to array indexing?
	return reinterpret_cast< const T* >( p + y * m_pitch );
}

template< typename T >
__inline__  __device__
T* KernelArray2D< T >::rowPointer( int y )
{
	ubyte* p = reinterpret_cast< ubyte* >( md_pPitchedPointer );

	// TODO: switch pointer arithmetic to array indexing?
	return reinterpret_cast< T* >( p + y * m_pitch );
}

template< typename T >
__inline__  __device__
int KernelArray2D< T >::width() const
{
	return m_width;
}

template< typename T >
__inline__  __device__
int KernelArray2D< T >::height() const
{
	return m_height;
}

template< typename T >
__inline__  __device__
size_t KernelArray2D< T >::pitch() const
{
	return m_pitch;
}

template< typename T >
__inline__  __device__
int2 KernelArray2D< T >::size() const
{
	return make_int2( m_width, m_height );
}

template< typename T >
__inline__  __device__
const T& KernelArray2D< T >::operator () ( int x, int y ) const
{
	return rowPointer( y )[ x ];
}

template< typename T >
__inline__  __device__
T& KernelArray2D< T >::operator () ( int x, int y )
{
	return rowPointer( y )[ x ];
}

template< typename T >
__inline__  __device__
const T& KernelArray2D< T >::operator [] ( const int2& xy ) const
{
	return rowPointer( xy.y )[ xy.x ];
}

template< typename T >
__inline__  __device__
T& KernelArray2D< T >::operator [] ( const int2& xy )
{
	return rowPointer( xy.y )[ xy.x ];
}
