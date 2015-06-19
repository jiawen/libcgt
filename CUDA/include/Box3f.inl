libcgt::cuda::Box3f::Box3f() :

	m_origin( make_float3( 0.f, 0.f, 0.f ) ),
	m_size( make_float3( 0.f, 0.f, 0.f ) )

{

}

libcgt::cuda::Box3f::Box3f( float left, float bottom, float back, float width, float height, float depth ) :

	m_origin( make_float3( left, bottom, back ) ),
	m_size( make_float3( width, height, depth ) )

{

}

libcgt::cuda::Box3f::Box3f( float width, float height, float depth ) :

	m_origin( make_float3( 0.f, 0.f, 0.f ) ),
	m_size( make_float3( width, height, depth ) )

{

}

libcgt::cuda::Box3f::Box3f( const float3& origin, const float3& size ) :

	m_origin( origin ),
	m_size( size )

{

}

libcgt::cuda::Box3f::Box3f( const float3& size ) :

	m_origin( make_float3( 0.f, 0.f, 0.f ) ),
	m_size( size )

{

}

float libcgt::cuda::Box3f::left() const
{
	return m_origin.x;
}

float libcgt::cuda::Box3f::right() const
{
	return m_origin.x + m_size.x;
}

float libcgt::cuda::Box3f::bottom() const
{
	return m_origin.y;
}

float libcgt::cuda::Box3f::top() const
{
	return m_origin.y + m_size.y;
}

float libcgt::cuda::Box3f::back() const
{
	return m_origin.z;
}

float libcgt::cuda::Box3f::front() const
{
	return m_origin.z + m_size.z;
}

float3 libcgt::cuda::Box3f::leftBottomBack() const
{
	return m_origin;
}

float3 libcgt::cuda::Box3f::rightTopFront() const
{
	return m_origin + m_size;
}

float3 libcgt::cuda::Box3f::center() const
{
	return m_origin + 0.5f * m_size;
}

void libcgt::cuda::Box3f::getCorners( float3 corners[8] ) const
{
	for( int i = 0; i < 8; ++i )
	{
		corners[ i ] =
			make_float3
			(
				( i & 1 ) ? m_origin.x : m_origin.x + m_size.x,
				( i & 2 ) ? m_origin.y : m_origin.y + m_size.y,
				( i & 4 ) ? m_origin.z : m_origin.z + m_size.z
			);
	}
}

bool libcgt::cuda::Box3f::contains( float x, float y, float z ) const
{
	return
	(
		( x >= m_origin.x ) &&
		( x < ( m_origin.x + m_size.x ) ) &&
		( y >= m_origin.y ) &&
		( y < ( m_origin.y + m_size.y ) ) &&
		( z >= m_origin.z ) &&
		( z < ( m_origin.z + m_size.z ) )
	);
}

bool libcgt::cuda::Box3f::contains( const float3& p ) const
{
	return contains( p.x, p.y, p.z );
}

// static
bool libcgt::cuda::Box3f::intersect( const libcgt::cuda::Box3f& r0, const libcgt::cuda::Box3f& r1 )
{
	libcgt::cuda::Box3f isect;
	return intersect( r0, r1, isect );
}

// static
bool libcgt::cuda::Box3f::intersect( const libcgt::cuda::Box3f& r0, const libcgt::cuda::Box3f& r1, libcgt::cuda::Box3f& intersection )
{
	float3 minimum = fmaxf( r0.leftBottomBack(), r1.leftBottomBack() );
	float3 maximum = fminf( r0.rightTopFront(), r1.rightTopFront() );

	if( minimum.x < maximum.x &&
		minimum.y < maximum.y )
	{
		intersection.m_origin = minimum;
		intersection.m_size = maximum - minimum;
		return true;
	}
	return false;
}
