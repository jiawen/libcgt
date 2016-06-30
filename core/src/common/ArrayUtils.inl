namespace libcgt { namespace core { namespace arrayutils {

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

template< typename S, typename T >
Array1DView< S > componentView( Array1DView< T > src, int componentOffsetBytes )
{
    // TODO: wrap const pointer correctly
    return Array1DView< S >( reinterpret_cast< uint8_t* >( src.pointer() ) + componentOffsetBytes,
        src.size(), src.stride() );
}

template< typename S, typename T >
Array2DView< S > componentView( Array2DView< T > src, int componentOffsetBytes )
{
    // TODO: wrap const pointer correctly
    return Array2DView< S >( reinterpret_cast< uint8_t* >( src.pointer() ) + componentOffsetBytes,
        src.size(), src.stride() );
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
    return Array2DView< T >( cornerPointer, range.size, view.stride() );
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

#if 0
template< typename TSrc, typename TDst >
bool ArrayUtils::map( Array1DView< TSrc > src, Array1DView< TDst > dst,
    std::function< TDst( TSrc ) > f )
#else
template< typename TSrc, typename TDst, typename Func >
bool map( Array1DView< const TSrc > src, Array1DView< TDst > dst, Func f )
#endif
{
    std::function< TDst( const TSrc ) > f2( f );

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
#if 0
        dst[ x ] = f( src[ x ] );
#else
        dst[ x ] = f2( src[ x ] );
#endif
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

template< typename T >
Array1DView< const T > readViewOf( const std::vector< T >& v )
{
    return Array1DView< const T >( v.data(), v.size() );
}

template< typename T >
Array1DView< T > writeViewOf( std::vector< T >& v )
{
    return Array1DView< T >(v.data(), v.size());
}


} } } // namespace arrayutils, core, libcgt

// static
template< typename T >
bool ArrayUtils::loadBinary( FILE* fp, std::vector< T >& output )
{
    int length;

    fread( &length, sizeof( int ), 1, fp );
    output.resize( length );

    fread( output.data(), sizeof( T ), length, fp );

    // TODO: error checking
    return true;
}

// static
template< typename T >
bool ArrayUtils::saveBinary( const std::vector< T >& input, const char* filename )
{
    FILE* fp = fopen( filename, "wb" );
    if( fp == nullptr )
    {
        return false;
    }

    bool succeeded = saveBinary( input, fp );
    fclose( fp );
    return succeeded;
}

// static
template< typename T >
bool ArrayUtils::saveBinary( const std::vector< T >& input, FILE* fp )
{
    // TODO: error check

    int length = static_cast< int >( input.size() );
    fwrite( &length, sizeof( int ), 1, fp );

    fwrite( input.data(), sizeof( T ), length, fp );

    return true;
}
