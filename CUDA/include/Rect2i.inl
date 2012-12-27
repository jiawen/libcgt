libcgt::cuda::Rect2i::Rect2i() :

	m_origin( make_int2( 0, 0 ) ),
	m_size( make_int2( 0, 0 ) )

{

}

libcgt::cuda::Rect2i::Rect2i( int left, int bottom, int width, int height ) :

	m_origin( make_int2( left, bottom ) ),
	m_size( make_int2( width, height ) )

{

}

libcgt::cuda::Rect2i::Rect2i( const int2& origin, const int2& size ) :

	m_origin( origin ),
	m_size( size )

{

}

int libcgt::cuda::Rect2i::left() const
{
	return m_origin.x;
}

int libcgt::cuda::Rect2i::right() const
{
	return m_origin.x + m_size.x;
}

int libcgt::cuda::Rect2i::bottom() const
{
	return m_origin.y;
}

int libcgt::cuda::Rect2i::top() const
{
	return m_origin.y + m_size.y;
}

int2 libcgt::cuda::Rect2i::bottomLeft() const
{
	return m_origin;
}

int2 libcgt::cuda::Rect2i::bottomRight() const
{
	return make_int2( right(), m_origin.y );
}

int2 libcgt::cuda::Rect2i::topLeft() const
{
	return make_int2( m_origin.x, top() );
}

int2 libcgt::cuda::Rect2i::topRight() const
{
	return m_origin + m_size;
}

int2 libcgt::cuda::Rect2i::origin() const
{
	return m_origin;
}

int2 libcgt::cuda::Rect2i::size() const
{
	return m_size;
}

int libcgt::cuda::Rect2i::area() const
{
	return m_size.x * m_size.y;
}

libcgt::cuda::Rect2i libcgt::cuda::Rect2i::flippedUD( int height ) const
{
	int2 origin;
	origin.x = m_origin.x;
	origin.y = height - topLeft().y;

	return Rect2i( origin, m_size );
}
