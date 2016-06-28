namespace libcgt { namespace cuda {

Rect2i::Rect2i( const int2& _size ) :
    size( _size )
{

}

Rect2i::Rect2i( const int2& _origin, const int2& _size ) :
    origin( _origin ),
    size( _size )
{

}

int Rect2i::left() const
{
    return origin.x;
}

int Rect2i::right() const
{
    return origin.x + size.x;
}

int Rect2i::bottom() const
{
    return origin.y;
}

int Rect2i::top() const
{
    return origin.y + size.y;
}

int2 Rect2i::bottomLeft() const
{
    return origin;
}

int2 Rect2i::bottomRight() const
{
    return make_int2( right(), origin.y );
}

int2 Rect2i::topLeft() const
{
    return make_int2( origin.x, top() );
}

int2 Rect2i::topRight() const
{
    return origin + size;
}

int Rect2i::area() const
{
    return size.x * size.y;
}

Rect2i flipY( const Rect2i& r, int height )
{
    int2 origin;
    origin.x = r.origin.x;
    origin.y = height - r.topLeft().y;

    return Rect2i( origin, r.size );
}

__inline__ __device__ __host__
Rect2i inset( const Rect2i& r, const int2& xy )
{
    return
    {
        r.origin + xy,
        r.size - 2 * xy
    };
}

} } // cuda, libcgt
