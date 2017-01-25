namespace libcgt { namespace core { namespace arrayutils {

template< typename TOut, typename TIn >
Array1DReadView< TOut > cast( Array1DReadView< TIn > src )
{
    static_assert( sizeof( TIn ) == sizeof( TOut ),
        "TIn and TOut must have the same size" );
    return Array1DReadView< TOut >( src.pointer(), src.size(), src.stride() );
}

template< typename TOut, typename TIn >
Array1DWriteView< TOut > cast( Array1DWriteView< TIn > src )
{
    static_assert( sizeof( TIn ) == sizeof( TOut ),
        "TIn and TOut must have the same size" );
    return Array1DWriteView< TOut >( src.pointer(), src.size(), src.stride() );
}

template< typename TOut, typename TIn >
Array2DReadView< TOut > cast( Array2DReadView< TIn > src )
{
    static_assert( sizeof( TIn ) == sizeof( TOut ),
        "TIn and TOut must have the same size" );
    return Array2DReadView< TOut >( src.pointer(), src.size(), src.stride() );
}

template< typename TOut, typename TIn >
Array2DWriteView< TOut > cast( Array2DWriteView< TIn > src )
{
    static_assert( sizeof( TIn ) == sizeof( TOut ),
        "TIn and TOut must have the same size" );
    return Array2DWriteView< TOut >( src.pointer(), src.size(), src.stride() );
}

template< typename TOut, typename TIn >
Array3DReadView< TOut > cast( Array3DReadView< TIn > src )
{
    static_assert( sizeof( TIn ) == sizeof( TOut ),
        "TIn and TOut must have the same size" );
    return Array3DReadView< TOut >( src.pointer(), src.size(), src.stride() );
}

template< typename TOut, typename TIn >
Array3DWriteView< TOut > cast( Array3DWriteView< TIn > src )
{
    static_assert( sizeof( TIn ) == sizeof( TOut ),
        "TIn and TOut must have the same size" );
    return Array3DWriteView< TOut >( src.pointer(), src.size(), src.stride() );
}

template< typename T >
bool copy( Array1DReadView< T > src, Array1DWriteView< T > dst )
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
bool copy( Array2DReadView< T > src, Array2DWriteView< T > dst )
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
        memcpy( dst.pointer(), src.pointer(),
            src.numElements() * src.elementStrideBytes() );
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
bool copy( Array3DReadView< T > src, Array3DWriteView< T > dst )
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
Array1DReadView< TOut > componentView( Array1DReadView< TIn > src,
    int componentOffsetBytes )
{
    return Array1DReadView< TOut >
    (
        reinterpret_cast< const uint8_t* >(
            src.pointer() ) + componentOffsetBytes,
            src.size(), src.stride()
    );
}

template< typename TOut, typename TIn >
Array1DWriteView< TOut > componentView( Array1DWriteView< TIn > src,
    int componentOffsetBytes )
{
    return Array1DWriteView< TOut >
    (
        reinterpret_cast< uint8_t* >( src.pointer() ) + componentOffsetBytes,
            src.size(), src.stride()
    );
}

template< typename TOut, typename TIn >
Array2DReadView< TOut > componentView( Array2DReadView< TIn > src,
    int componentOffsetBytes )
{
    return Array2DReadView< TOut >
    (
        reinterpret_cast< const uint8_t* >(
            src.pointer() ) + componentOffsetBytes,
            src.size(), src.stride()
    );
}

template< typename TOut, typename TIn >
Array2DWriteView< TOut > componentView( Array2DWriteView< TIn > src,
    int componentOffsetBytes )
{
    return Array2DWriteView< TOut >
    (
        reinterpret_cast< uint8_t* >( src.pointer() ) + componentOffsetBytes,
            src.size(), src.stride()
    );
}

template< typename TOut, typename TIn >
Array3DReadView< TOut > componentView( Array3DReadView< TIn > src,
    int componentOffsetBytes )
{
    return Array3DReadView< TOut >
    (
        reinterpret_cast< const uint8_t* >
            ( src.pointer() ) + componentOffsetBytes,
            src.size(), src.stride()
    );
}

template< typename TOut, typename TIn >
Array3DWriteView< TOut > componentView( Array3DWriteView< TIn > src,
    int componentOffsetBytes )
{
    return Array3DWriteView< TOut >
    (
        reinterpret_cast< uint8_t* >( src.pointer() ) + componentOffsetBytes,
            src.size(), src.stride()
    );
}

template< typename T >
Array1DWriteView< T > crop( Array1DWriteView< T > view, int x )
{
    return crop( view, { x, view.width() - x } );
}

template< typename T >
Array1DWriteView< T > crop( Array1DWriteView< T > view, const Range1i& range )
{
    T* cornerPointer = view.elementPointer( range.origin );
    return Array1DWriteView< T >( cornerPointer, range.size, view.stride() );
}

template< typename T >
Array2DReadView< T > crop( Array2DReadView< T > view, const Vector2i& xy )
{
    return crop( view, { xy, view.size() - xy } );
}

template< typename T >
Array2DWriteView< T > crop( Array2DWriteView< T > view, const Vector2i& xy )
{
    return crop( view, { xy, view.size() - xy } );
}

template< typename T >
Array2DReadView< T > crop( Array2DReadView< T > view, const Rect2i& rect )
{
    T* cornerPointer = view.elementPointer( rect.origin );
    return Array2DReadView< T >( cornerPointer, rect.size, view.stride() );
}

template< typename T >
Array2DWriteView< T > crop( Array2DWriteView< T > view, const Rect2i& rect )
{
    T* cornerPointer = view.elementPointer( rect.origin );
    return Array2DWriteView< T >( cornerPointer, rect.size, view.stride() );
}

template< typename T >
Array3DWriteView< T > crop( Array3DWriteView< T > view, const Vector3i& xyz )
{
    return crop( view, { xyz, view.size() - xyz } );
}

template< typename T >
Array3DWriteView< T > crop( Array3DWriteView< T > view, const Box3i& box )
{
    T* cornerPointer = view.elementPointer( box.origin );
    return Array3DWriteView< T >( cornerPointer, box.size, view.stride() );
}

template< typename T >
bool clear( Array2DWriteView< T > view )
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
            memset( view.rowPointer( y ), 0,
                view.elementStrideBytes() * view.width() );
        }
        return true;
    }
    return false;
}

template< typename T >
bool fill( Array1DWriteView< T > view, const T& value )
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
bool fill( Array2DWriteView< T > view, const T& value )
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
Array1DReadView< T > flipX( Array1DReadView< T > src )
{
    return Array1DWriteView< T >
    (
        src.elementPointer( src.length() - 1 ),
        src.length(),
        -src.elementStrideBytes()
    );
}

template< typename T >
Array1DWriteView< T > flipX( Array1DWriteView< T > src )
{
    return Array1DWriteView< T >
    (
        src.elementPointer( src.length() - 1 ),
        src.length(),
        -src.elementStrideBytes()
    );
}

template< typename T >
Array2DReadView< T > flipX( Array2DReadView< T > src )
{
    Vector2i stride = src.stride();
    stride.x = -stride.x;

    return Array2DReadView< T >
    (
        src.elementPointer( { src.width() - 1, 0 } ),
        src.size(),
        stride
    );
}

template< typename T >
Array2DWriteView< T > flipX( Array2DWriteView< T > src )
{
    Vector2i stride = src.stride();
    stride.x = -stride.x;

    return Array2DWriteView< T >
    (
        src.elementPointer( { src.width() - 1, 0 } ),
        src.size(),
        stride
    );
}

template< typename T >
Array2DReadView< T > flipY( Array2DReadView< T > src )
{
    Vector2i stride = src.stride();
    stride.y = -stride.y;

    return Array2DReadView< T >
    (
        src.rowPointer( src.height() - 1 ),
        src.size(),
        stride
    );
}

template< typename T >
Array2DWriteView< T > flipY( Array2DWriteView< T > src )
{
    Vector2i stride = src.stride();
    stride.y = -stride.y;

    return Array2DWriteView< T >
    (
        src.rowPointer( src.height() - 1 ),
        src.size(),
        stride
    );
}

template< typename T >
void flipYInPlace( Array2DWriteView< T > v )
{
    Array1D< T > tmp( v.width() );
    for( int y = 0; y < v.height() / 2; ++y )
    {
        // Copy row y into tmp.
        copy< T >( v.row( y ), tmp );
        // Copy row (height - y - 1) into y.
        copy< T >( v.row( v.height() - y - 1 ), v.row( y ) );
        // Copy tmp into row (height - y - 1).
        copy< T >( tmp, v.row( v.height() - y - 1 ) );
    }
}

template< typename T >
Array2DReadView< T > transpose( Array2DReadView< T > src )
{
    Vector2i size = src.size().yx();
    Vector2i stride = src.stride().yx();

    return Array2DReadView< T >
    (
        src.pointer(),
        size,
        stride
    );
}

template< typename T >
Array2DWriteView< T > transpose( Array2DWriteView< T > src )
{
    Vector2i size = src.size().yx();
    Vector2i stride = src.stride().yx();

    return Array2DWriteView< T >
    (
        src.pointer(),
        size,
        stride
    );
}

template< typename T >
Array3DWriteView< T > flipX( Array3DWriteView< T > src )
{
    Vector3i stride = src.stride();
    stride.x = -stride.x;

    return Array3DWriteView< T >
    (
        src.elementPointer( { src.width() - 1, 0, 0 } ),
        src.size(),
        stride
    );
}

template< typename T >
Array3DWriteView< T > flipY( Array3DWriteView< T > src )
{
    Vector3i stride = src.stride();
    stride.y = -stride.y;

    return Array3DWriteView< T >
    (
        src.rowPointer( { src.height() - 1, 0 } ),
        src.size(),
        stride
    );
}

template< typename T >
Array3DWriteView< T > flipZ( Array3DWriteView< T > src )
{
    Vector3i stride = src.stride();
    stride.z = -stride.z;

    return Array3DWriteView< T >
    (
        src.slicePointer( src.depth() - 1 ),
        src.size(),
        stride
    );
}

template< typename T >
Array1DWriteView< T > head( Array1DWriteView< T > src, size_t n )
{
    if( src.size() < n )
    {
        return Array1DWriteView< T >();
    }

    return Array1DWriteView< T >( src.pointer(), n, src.stride() );
}

template< typename T >
Array1DWriteView< T > tail( Array1DWriteView< T > src, size_t n )
{
    if( src.size() < n )
    {
        return Array1DWriteView< T >();
    }

    T* p = src.elementPointer( src.size() - n );
    return Array1DWriteView< T >( p, n, src.stride() );
}

template< typename TSrc, typename TDst, typename Func >
bool map( Array1DReadView< TSrc > src, Array1DWriteView< TDst > dst, Func f )
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
bool map( Array2DReadView< TSrc > src, Array2DWriteView< TDst > dst, Func f )
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
bool map( Array3DReadView< TSrc > src, Array3DWriteView< TDst > dst, Func f )
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
bool mapIndexed( Array1DReadView< TSrc > src, Array1DWriteView< TDst > dst,
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
bool mapIndexed( Array2DReadView< TSrc > src, Array2DWriteView< TDst > dst,
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

template< typename TSrc, typename TDst, typename Func >
bool mapIndexed( Array3DReadView< TSrc > src, Array3DWriteView< TDst > dst,
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

    for( int z = 0; z < src.depth(); ++z )
    {
        for( int y = 0; y < src.height(); ++y )
        {
            for( int x = 0; x < src.width(); ++x )
            {
                dst[ { x, y, z } ] = f( { x, y, z }, src[ { x, y, z } ] );
            }
        }
    }

    return true;
}

template< typename T >
Array1DWriteView< T > reshape( Array2DWriteView< T > src )
{
    if( src.isNull() || !src.rowsArePacked() )
    {
        return Array1DWriteView< T >();
    }

    return Array1DWriteView< T >
    (
        src.pointer(),
        src.numElements(),
        src.elementStrideBytes()
    );
}

template< typename T >
Array1DWriteView< T > reshape( Array3DWriteView< T > src )
{
    if( src.isNull() || !src.rowsArePacked() || !src.slicesArePacked() )
    {
        return Array1DWriteView< T >();
    }

    return Array1DWriteView< T >
    (
        src.pointer(),
        src.numElements(),
        src.elementStrideBytes()
    );
}

template< typename T >
Array1DReadView< T > readViewOf( const std::vector< T >& v, size_t offset )
{
    return Array1DReadView< T >( v.data() + offset, v.size() - offset );
}

template< typename T >
Array1DWriteView< T > writeViewOf( std::vector< T >& v, size_t offset )
{
    return Array1DWriteView< T >( v.data() + offset, v.size() - offset );
}

} } } // namespace arrayutils, core, libcgt
