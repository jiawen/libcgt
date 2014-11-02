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
template<>
bool ArrayUtils::fill( Array2DView< uint8_t > view, const uint8_t& value )
{
	if( view.isNull() )
	{
		return false;
	}

	if( view.packed() )
	{
		memset( view.pointer(), value, view.bytesSpanned() );
		return true;
	}

	if( view.elementsArePacked() )
	{
		// Fill w bytes at a time.
		int w = view.width();
		for( int y = 0; y < view.height(); ++y )
		{
			memset( view.rowPointer( y ), value, w );
		}
		return true;
	}

	// Nothing is packed, iterate.
	int ne = view.numElements();
	for( int k = 0; k < ne; ++k )
	{
		view[ k ] = value;
	}
	return true;
}

// static
template< typename T >
Array1DView< T > ArrayUtils::flipLR( Array1DView< T > view )
{
	return Array1DView< T >
	(
		&( view( view.length() - 1, 0 ) ),
		view.length(),
		-view.elementStrideBytes()
	);
}

// static
template< typename T >
Array2DView< T > ArrayUtils::flipLR( Array2DView< T > view )
{
	return Array2DView< T >
	(
		&( view( view.width() - 1, 0 ) ),
		view.size(),
		-view.elementStrideBytes(),
		view.rowStrideBytes()
	);
}

// static
template< typename T >
Array2DView< T > ArrayUtils::flipUD( Array2DView< T > view )
{
	return Array2DView< T >
	(
		view.rowPointer( view.height() - 1 ),
		view.size(),
		view.elementStrideBytes(),
		-view.rowStrideBytes()
	);
}

// static
template< typename T >
Array1DView< T > ArrayUtils::crop( Array1DView< T > view, int x, int width )
{	
	return Array1DView< T >( view.elementPointer( x ), width, view.elementStrideBytes() );
}

// static
template< typename T >
Array2DView< T > ArrayUtils::crop( Array2DView< T > view, int x, int y, int width, int height )
{	
	return Array2DView< T >( view.elementPointer( x, y ), width, height, view.elementStrideBytes(), view.rowStrideBytes() );
}

// static
template< typename T >
Array3DView< T > ArrayUtils::crop( Array3DView< T > view, int x, int y, int z, int width, int height, int depth )
{
	return Array3DView< T >( view.elementPointer( x, y, z ), width, height, depth, view.elementStrideBytes(), view.rowStrideBytes(), view.slicePitchBytes() );
}

// static
template< typename T >
bool ArrayUtils::copy( Array1DView< T > src, Array1DView< T > dst )
{
	if( src.isNull() || dst.isNull() )
	{
		return false;
	}

	if( src.length() != dst.length() )
	{
		return false;
	}

	if( src.packed() && dst.packed() )
	{
		memcpy( dst.pointer(), src.pointer(), src.bytesSpanned() );
	}	
	else
	{
		for( int x = 0; x < src.length(); ++x )
		{
			dst[ x ] = src[ x ];
		}
	}

	return true;
}

// static
template< typename T >
bool ArrayUtils::copy( Array2DView< T > src, Array2DView< T > dst )
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
		memcpy( dst.pointer(), src.pointer(), src.bytesSpanned() );
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
				dst( x, y ) = src( x, y );
			}
		}
	}

	return true;
}
