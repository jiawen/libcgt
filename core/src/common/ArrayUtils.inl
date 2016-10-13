namespace libcgt { namespace core { namespace arrayutils {

template< typename TOut, typename TIn >
Array1DView< TOut > cast( Array1DView< TIn > src )
{
    static_assert( sizeof( TIn ) == sizeof( TOut ),
        "TIn and TOut must have the same size" );
    return Array1DView< TOut >( src.pointer(), src.size(), src.stride() );
}

template< typename TOut, typename TIn >
Array2DView< TOut > cast( Array2DView< TIn > src )
{
    static_assert( sizeof( TIn ) == sizeof( TOut ),
        "TIn and TOut must have the same size" );
    return Array2DView< TOut >( src.pointer(), src.size(), src.stride() );
}

template< typename TOut, typename TIn >
Array3DView< TOut > cast( Array3DView< TIn > src )
{
    static_assert( sizeof( TIn ) == sizeof( TOut ),
        "TIn and TOut must have the same size" );
    return Array3DView< TOut >( src.pointer(), src.size(), src.stride() );
}

template< typename T >
bool copy( Array1DView< const T > src, Array1DView< T > dst )
{
    if( src.isNull() || dst.isNull() )
    {
        return false;
    }

    if( src.size() != dst.size() )
    {
        return false;
    }

    // Both views are packed, do a single memcpy.
    if( src.packed() && dst.packed() )
    {
        memcpy( dst.pointer(), src.pointer(), src.size() * src.stride() );
    }
    // Otherwise, iterate.
    else
    {
        for( int x = 0; x < src.size(); ++x )
        {
            dst[ x ] = src[ x ];
        }
    }

    return true;
}

template< typename T >
bool copy( Array2DView< const T > src, Array2DView< T > dst )
{
    if( src.isNull() || dst.isNull() )
    {
        return false;
    }

    if( src.size() != dst.size() )
    {
        return false;
    }

    // Both views are packed, do a single memcpy.
    if( src.packed() && dst.packed() )
    {
        memcpy( dst.pointer(), src.pointer(), src.numElements() * src.elementStrideBytes() );
    }
    // Otherwise, copy row by row.
    else
    {
        for( int y = 0; y < src.height(); ++y )
        {
            copy( src.row( y ), dst.row( y ) );
        }
    }

    return true;
}

template< typename T >
bool copy( Array3DView< const T > src, Array3DView< T > dst )
{
    if( src.isNull() || dst.isNull() )
    {
        return false;
    }

    if( src.size() != dst.size() )
    {
        return false;
    }

    // Both views are packed, do a single memcpy.
    if( src.packed() && dst.packed() )
    {
        memcpy( dst.pointer(), src.pointer(), src.numElements() * src.elementStrideBytes() );
    }
    // Otherwise, copy slice by slice.
    else
    {
        for( int z = 0; z < src.depth(); ++z )
        {
            copy( src.row( z ), dst.row( z ) );
        }
    }

    return true;
}

template< typename TOut, typename TIn >
Array1DView< TOut > componentView( Array1DView< TIn > src, int componentOffsetBytes )
{
    return Array1DView< TOut >
    (
        reinterpret_cast< typename Array1DView< TIn >::UInt8Pointer >
            ( src.pointer() ) + componentOffsetBytes,
        src.size(), src.stride()
    );
}

template< typename TOut, typename TIn >
Array2DView< TOut > componentView( Array2DView< TIn > src, int componentOffsetBytes )
{
    return Array2DView< TOut >
    (
        reinterpret_cast< typename Array2DView< TIn >::UInt8Pointer >
            ( src.pointer() ) + componentOffsetBytes,
        src.size(), src.stride()
    );
}

template< typename TOut, typename TIn >
Array3DView< TOut > componentView( Array3DView< TIn > src, int componentOffsetBytes )
{
    return Array3DView< TOut >
    (
        reinterpret_cast< typename Array3DView< TIn >::UInt8Pointer >
            ( src.pointer() ) + componentOffsetBytes,
        src.size(), src.stride()
    );
}

template< typename T >
Array1DView< T > crop( Array1DView< T > view, int x )
{
    return crop( view, { x, view.width() - x } );
}

template< typename T >
Array1DView< T > crop( Array1DView< T > view, const Range1i& range )
{
    T* cornerPointer = view.elementPointer( range.origin );
    return Array1DView< T >( cornerPointer, range.size, view.stride() );
}

template< typename T >
Array2DView< T > crop( Array2DView< T > view, const Vector2i& xy )
{
    return crop( view, { xy, view.size() - xy } );
}

template< typename T >
Array2DView< T > crop( Array2DView< T > view, const Rect2i& rect )
{
    T* cornerPointer = view.elementPointer( rect.origin );
    return Array2DView< T >( cornerPointer, rect.size, view.stride() );
}

template< typename T >
Array3DView< T > crop( Array3DView< T > view, const Vector3i& xyz )
{
    return crop( view, { xyz, view.size() - xyz } );
}

template< typename T >
Array3DView< T > crop( Array3DView< T > view, const Box3i& box )
{
    T* cornerPointer = view.elementPointer( box.origin );
    return Array3DView< T >( cornerPointer, box.size, view.stride() );
}

template< typename T >
bool clear( Array2DView< T > view )
{
    if( view.isNull() )
    {
        return false;
    }
    if( view.packed() )
    {
        memset( view, 0, view.rowStrideBytes() * view.height() );
        return true;
    }
    else if( view.elementsArePacked() )
    {
        for( int y = 0; y < view.height(); ++y )
        {
            memset( view.rowPointer( y ), 0, view.elementStrideBytes() * view.width() );
        }
        return true;
    }
    return false;
}

template< typename T >
bool fill( Array1DView< T > view, const T& value )
{
    if( view.isNull() )
    {
        return false;
    }

    size_t ne = view.numElements();
    for( size_t k = 0; k < ne; ++k )
    {
        view[ k ] = value;
    }
    return true;
}

template< typename T >
bool fill( Array2DView< T > view, const T& value )
{
    if( view.isNull() )
    {
        return false;
    }

    int ne = view.numElements();
    for( int k = 0; k < ne; ++k )
    {
        view[ k ] = value;
    }
    return true;
}

template< typename T >
Array1DView< T > flipX( Array1DView< T > src )
{
    return Array1DView< T >
    (
        src.elementPointer( src.length() - 1 ),
        src.length(),
        -src.elementStrideBytes()
    );
}

template< typename T >
Array2DView< T > flipX( Array2DView< T > src )
{
    Vector2i stride = src.stride();
    stride.x = -stride.x;

    return Array2DView< T >
    (
        src.elementPointer( { src.width() - 1, 0 } ),
        src.size(),
        stride
    );
}

template< typename T >
Array2DView< T > flipY( Array2DView< T > src )
{
    Vector2i stride = src.stride();
    stride.y = -stride.y;

    return Array2DView< T >
    (
        src.rowPointer( src.height() - 1 ),
        src.size(),
        stride
    );
}

// TODO: ugh, Array1DView< const T > is a big hack. Reconsider this design.
template< typename T >
void flipYInPlace( Array2DView< T > v )
{
    Array1D< T > tmp( v.width() );
    for( int y = 0; y < v.height() / 2; ++y )
    {
        // Copy row y into tmp.
        copy( Array1DView< const T >( v.row( y ) ), tmp.writeView() );
        // Copy row (height - y - 1) into y.
        copy( Array1DView< const T >( v.row( v.height() - y - 1 ) ),
            v.row( y ) );
        // Copy tmp into row (height - y - 1).
        copy( tmp.readView(), v.row( v.height() - y - 1 ) );
    }
}

template< typename T >
Array2DView< T > transpose( Array2DView< T > src )
{
    Vector2i size = src.size().yx();
    Vector2i stride = src.stride().yx();

    return Array2DView< T >
    (
        src.pointer(),
        size,
        stride
    );
}

template< typename T >
Array3DView< T > flipX( Array3DView< T > src )
{
    Vector3i stride = src.stride();
    stride.x = -stride.x;

    return Array3DView< T >
    (
        src.elementPointer( { src.width() - 1, 0 } ),
        src.size(),
        stride
    );
}

template< typename T >
Array3DView< T > flipY( Array3DView< T > src )
{
    Vector3i stride = src.stride();
    stride.y = -stride.y;

    return Array3DView< T >
    (
        src.rowPointer( src.height() - 1 ),
        src.size(),
        stride
    );
}

template< typename T >
Array3DView< T > flipZ( Array3DView< T > src )
{
    Vector3i stride = src.stride();
    stride.z = -stride.z;

    return Array2DView< T >
    (
        src.slicePointer( src.depth() - 1 ),
        src.size(),
        stride
    );
}

template< typename T >
Array1DView< T > head( Array1DView< T > src, size_t n )
{
    if( src.size() < n )
    {
        return Array1DView< T >();
    }

    return Array1DView< T >( src.pointer(), n, src.stride() );
}

template< typename T >
Array1DView< T > tail( Array1DView< T > src, size_t n )
{
    if( src.size() < n )
    {
        return Array1DView< T >();
    }

    T* p = src.elementPointer( src.size() - n );
    return Array1DView< T >( p, n, src.stride() );
}

template< typename TSrc, typename TDst, typename Func >
bool map( Array1DView< const TSrc > src, Array1DView< TDst > dst, Func f )
{
    if( src.isNull() || dst.isNull() )
    {
        return false;
    }

    if( src.size() != dst.size() )
    {
        return false;
    }

    for( int x = 0; x < src.size(); ++x )
    {
        dst[ x ] = f( src[ x ] );
    }

    return true;
}

template< typename TSrc, typename TDst, typename Func >
bool map( Array2DView< const TSrc > src, Array2DView< TDst > dst, Func f )
{
    if( src.isNull() || dst.isNull() )
    {
        return false;
    }

    if( src.size() != dst.size() )
    {
        return false;
    }

    for( int y = 0; y < src.height(); ++y )
    {
        for( int x = 0; x < src.width(); ++x )
        {
            dst[ { x, y } ] = f( src[ { x, y } ] );
        }
    }

    return true;
}

template< typename TSrc, typename TDst, typename Func >
bool map( Array3DView< const TSrc > src, Array3DView< TDst > dst, Func f )
{
    if( src.isNull() || dst.isNull() )
    {
        return false;
    }

    if( src.size() != dst.size() )
    {
        return false;
    }

    for( int z = 0; z < src.height(); ++z )
    {
        for( int y = 0; y < src.height(); ++y )
        {
            for( int x = 0; x < src.width(); ++x )
            {
                dst[ { x, y, z } ] = f( src[ { x, y, z } ] );
            }
        }
    }

    return true;
}

template< typename TSrc, typename TDst, typename Func >
bool mapIndexed( Array1DView< const TSrc > src, Array1DView< TDst > dst,
    Func f )
{
    if( src.isNull() || dst.isNull() )
    {
        return false;
    }

    if( src.size() != dst.size() )
    {
        return false;
    }

    for( int x = 0; x < src.width(); ++x )
    {
        dst[ x ] = f( x, src[ x ] );
    }

    return true;
}

template< typename TSrc, typename TDst, typename Func >
bool mapIndexed( Array2DView< const TSrc > src, Array2DView< TDst > dst,
    Func f )
{
    if( src.isNull() || dst.isNull() )
    {
        return false;
    }

    if( src.size() != dst.size() )
    {
        return false;
    }

    for( int y = 0; y < src.height(); ++y )
    {
        for( int x = 0; x < src.width(); ++x )
        {
            dst[ { x, y } ] = f( { x, y }, src[ { x, y } ] );
        }
    }

    return true;
}

template< typename T >
Array1DView< T > reshape( Array2DView< T > src )
{
    if( src.isNull() || !src.rowsArePacked() )
    {
        return Array1DView< T >();
    }

    return Array1DView< T >
    (
        src.pointer(),
        src.numElements(),
        src.elementStrideBytes()
    );
}

template< typename T >
Array1DView< T > reshape( Array3DView< T > src )
{
    if( src.isNull() || !src.rowsArePacked() || !src.slicesArePacked() )
    {
        return Array1DView< T >();
    }

    return Array1DView< T >
    (
        src.pointer(),
        src.numElements(),
        src.elementStrideBytes()
    );
}

template< typename T >
Array1DView< const T > readViewOf( const std::vector< T >& v )
{
    return Array1DView< const T >( v.data(), v.size() );
}

template< typename T >
Array1DView< T > writeViewOf( std::vector< T >& v )
{
    return Array1DView< T >( v.data(), v.size() );
}

} } } // namespace arrayutils, core, libcgt
