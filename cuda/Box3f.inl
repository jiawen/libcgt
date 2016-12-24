namespace libcgt { namespace cuda {

Box3f::Box3f( const float3& size ) :
    m_size( size )
{

}

Box3f::Box3f( const int3& size ) :
    m_size( make_float3( size.x, size.y, size.z ) )
{

}

Box3f::Box3f( const float3& origin, const float3& size ) :
    m_origin( origin ),
    m_size( size )
{

}

Box3f::Box3f( const int3& origin, const int3& size ) :
    m_origin( make_float3( origin.x, origin.y, origin.z ) ),
    m_size( make_float3( size.x, size.y, size.z ) )
{

}

float Box3f::left() const
{
    return m_origin.x;
}

float Box3f::right() const
{
    return m_origin.x + m_size.x;
}

float Box3f::bottom() const
{
    return m_origin.y;
}

float Box3f::top() const
{
    return m_origin.y + m_size.y;
}

float Box3f::back() const
{
    return m_origin.z;
}

float Box3f::front() const
{
    return m_origin.z + m_size.z;
}

float3 Box3f::minimum() const
{
    return m_origin;
}

float3 Box3f::maximum() const
{
    return m_origin + m_size;
}

float3 Box3f::center() const
{
    return m_origin + 0.5f * m_size;
}

void Box3f::getCorners( float3 corners[8] ) const
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

bool Box3f::contains( float x, float y, float z ) const
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

bool Box3f::contains( const float3& p ) const
{
    return contains( p.x, p.y, p.z );
}

// static
bool Box3f::intersect( const Box3f& r0, const Box3f& r1 )
{
    Box3f isect;
    return intersect( r0, r1, isect );
}

// static
bool Box3f::intersect( const Box3f& r0, const Box3f& r1, Box3f& intersection )
{
    float3 minimum = fmaxf( r0.minimum(), r1.minimum() );
    float3 maximum = fminf( r0.maximum(), r1.maximum() );

    if( minimum.x < maximum.x &&
        minimum.y < maximum.y )
    {
        intersection.m_origin = minimum;
        intersection.m_size = maximum - minimum;
        return true;
    }
    return false;
}

__inline__ __host__ __device__
bool intersectLine( const float3& origin, const float3& direction,
    const Box3f& box,
    float& tNear, float& tFar )
{
    // Compute t to each face.
    float3 rcpDir = 1.0f / direction;

    // Intersect the three "bottom" faces (min of the box).
    float3 tBottom = rcpDir * (box.minimum() - origin);
    // Intersect the three "top" faces (max of the box).
    float3 tTop = rcpDir * (box.maximum() - origin);

    // Find the smallest and largest distances along each axis.
    float3 tMin = fminf( tBottom, tTop );
    float3 tMax = fmaxf( tBottom, tTop );

    // tNear is the largest tMin
    tNear = libcgt::cuda::math::maximum(tMin);

    // tFar is the smallest tMax
    tFar = libcgt::cuda::math::minimum(tMax);

    return tFar > tNear;
}

} } // cuda, libcgt
