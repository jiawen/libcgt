template< typename T >
// static
bool ArrayUtils::loadBinary( FILE* fp, std::vector< T >& output )
{
	int length;

	fread( &length, sizeof( int ), 1, fp );
	output.resize( length );

	fread( output.data(), sizeof( T ), length, fp );

	// TODO: error checking
	return true;
}

template< typename T >
// static
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

template< typename T >
// static
bool ArrayUtils::saveBinary( const std::vector< T >& input, FILE* fp )
{
	// TODO: error check

	int length = static_cast< int >( input.size() );
	fwrite( &length, sizeof( int ), 1, fp );

	fwrite( input.data(), sizeof( T ), length, fp );

	return true;
}

// static
template< typename T >
bool ArrayUtils::clear( Array2DView< T > view )
{
    if( view.isNull() )
    {
        return false;
    }
    if( view.packed() )
    {
        memset( view, 0, view.rowStrideBytes() * view.height() );
    }
    else if( view.elementsArePacked() )
    {
        for( int y = 0; y < view.height(); ++y )
        {
            memset( view.rowPointer( y ), 0, view.elementStrideBytes() * view.width() );
        }
    }
}

// static
template< typename T >
bool ArrayUtils::fill( Array1DView< T > view, const T& value )
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

// static
template< typename T >
bool ArrayUtils::fill( Array2DView< T > view, const T& value )
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

// static
template< typename T >
Array1DView< T > ArrayUtils::flippedLeftRightView( Array1DView< T > view )
{
    return Array1DView< T >
    (
        view.elementPointer( view.length() - 1 ),
        view.length(),
        -view.elementStrideBytes()
    );
}

// static
template< typename T >
Array2DView< T > ArrayUtils::flippedLeftRightView( Array2DView< T > view )
{
	return Array2DView< T >
	(
        view.elementPointer( view.length() - 1, 0 ),
		view.size(),
        { -view.elementStrideBytes(), view.rowStrideBytes() }
	);
}

// static
template< typename T >
Array2DView< T > ArrayUtils::flippedUpDownView( Array2DView< T > view )
{
    return Array2DView< T >
    (
        view.rowPointer( view.height( ) - 1 ),
        view.size( ),
        { view.elementStrideBytes(), -view.rowStrideBytes() }
    );
}

// static
template< typename T >
Array2DView< T > ArrayUtils::croppedView( Array2DView< T > view, const Vector2i& xy )
{
    return croppedView( view, xy, { view.width() - xy.x, view.height() - xy.y } );
}

// static
template< typename T >
Array2DView< T > ArrayUtils::croppedView( Array2DView< T > view, const Rect2i& rect )
{
    T* cornerPointer = view.elementPointer( rect.origin() );
    return Array2DView< T >( cornerPointer, size, view.strides() );
}

// static
template< typename T >
Array3DView< T > ArrayUtils::croppedView( Array3DView< T > view, const Vector3i& xyz )
{
	return croppedView( view, xyz, view.width() - xyz.x, view.height() - xyz.y, view.depth() - xyz.z );
}

// static
template< typename T >
Array3DView< T > ArrayUtils::croppedView( Array3DView< T > view, const Box3i& box )
{
    T* cornerPointer = view.elementPointer( box.origin() );
    return Array3DView< T >( cornerPointer, size, view.strides() );
}

// static
template< typename S, typename T >
Array1DView< S > ArrayUtils::componentView( Array1DView< T > view, int componentOffsetBytes )
{
    return Array1DView< S >( reinterpret_cast< uint8_t* >( view.pointer() ) + componentOffsetBytes,
        view.size(), view.stride() );
}

// static
template< typename T >
bool ArrayUtils::copy( Array1DView< const T > src, Array1DView< T > dst )
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


// static
template< typename T >
bool ArrayUtils::copy( Array2DView< const T > src, Array2DView< T > dst )
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
	// Elements are packed within each row, do a memcpy per row.
	else if( src.elementsArePacked() && dst.elementsArePacked() )
	{
		for( int y = 0; y < src.height(); ++y )
		{
			memcpy( dst.rowPointer( y ), src.rowPointer( y ), src.width() * src.elementStrideBytes() );
		}
	}
	// Otherwise, iterate.
	else
	{
		for( int y = 0; y < src.height(); ++y )
		{
			for( int x = 0; x < src.width(); ++x )
            {
                dst[ { x, y } ] = src[ { x, y } ];
			}
		}
	}

	return true;
}

// static
#if 0
template< typename TSrc, typename TDst >
bool ArrayUtils::map( Array1DView< TSrc > src, Array1DView< TDst > dst,
    std::function< TDst( TSrc ) > f )
#else
template< typename TSrc, typename TDst, typename Func >
bool ArrayUtils::map( Array1DView< TSrc > src, Array1DView< TDst > dst,
    Func f )
#endif
{
    std::function< TDst( TSrc ) > f2( f );

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