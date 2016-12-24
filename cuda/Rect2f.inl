namespace libcgt { namespace cuda {

Rect2f::Rect2f() :

    m_origin( make_float2( 0.f, 0.f ) ),
    m_size( make_float2( 0.f, 0.f ) )

{

}

Rect2f::Rect2f( float left, float bottom, float width, float height ) :

    m_origin( make_float2( left, bottom ) ),
    m_size( make_float2( width, height ) )

{

}

Rect2f::Rect2f( float width, float height ) :

    m_origin( make_float2( 0.f, 0.f ) ),
    m_size( make_float2( width, height ) )

{

}

Rect2f::Rect2f( const float2& origin, const float2& size ) :

    m_origin( origin ),
    m_size( size )

{

}

Rect2f::Rect2f( const float2& size ) :

    m_origin( make_float2( 0.f, 0.f ) ),
    m_size( size )

{

}

float Rect2f::left() const
{
    return m_origin.x;
}

float Rect2f::right() const
{
    return m_origin.x + m_size.x;
}

float Rect2f::bottom() const
{
    return m_origin.y;
}

float Rect2f::top() const
{
    return m_origin.y + m_size.y;
}

float2 Rect2f::bottomLeft() const
{
    return m_origin;
}

float2 Rect2f::bottomRight() const
{
    return make_float2( m_origin.x + m_size.x, m_origin.y );
}

float2 Rect2f::topLeft() const
{
    return make_float2( m_origin.x, m_origin.y + m_size.y );
}

float2 Rect2f::topRight() const
{
    return m_origin + m_size;
}

float2 Rect2f::origin() const
{
    return m_origin;
}

float2 Rect2f::size() const
{
    return m_size;
}

float Rect2f::area() const
{
    return m_size.x * m_size.y;
}

Rect2i Rect2f::enlargedToInt() const
{
    int2 minimum = floorToInt( bottomLeft() );
    int2 maximum = floorToInt( topRight() );

    // size does not need a +1:
    // say min is 1.1 and max is 3.6
    // then floor( min ) = 1 and ceil( 3.6 ) is 4
    // hence, we want indices 1, 2, 3, which has size 3 = 4 - 1
    int2 size = maximum - minimum;

    return Rect2i( minimum, size );
}

// static
bool Rect2f::intersect( const Rect2f& r0, const Rect2f& r1 )
{
    Rect2f isect;
    return intersect( r0, r1, isect );
}

// static
bool Rect2f::intersect( const Rect2f& r0, const Rect2f& r1, Rect2f& intersection )
{
    float2 minimum = fmaxf( r0.bottomLeft(), r1.bottomLeft() );
    float2 maximum = fminf( r0.topRight(), r1.topRight() );

    if( minimum.x < maximum.x &&
        minimum.y < maximum.y )
    {
        intersection.m_origin = minimum;
        intersection.m_size = maximum - minimum;
        return true;
    }
    return false;
}

} } // cuda, libcgt
