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
bool ArrayUtils::copy( Array2DView< T > src, Array2DView< T > dst )
{
	if( src.rowPointer( 0 ) == nullptr ||
		dst.rowPointer( 0 ) == nullptr )
	{
		return false;
	}

	if( src.size() != dst.size() )
	{
		return false;
	}

	if( src.packed() && dst.packed() )
	{
		memcpy( dst.rowPointer( 0 ), src.rowPointer( 0 ), src.rowPitchBytes() * src.height() );
	}
	else if( src.elementsArePacked() && dst.elementsArePacked() )
	{
		for( int y = 0; y < src.height(); ++y )
		{
			memcpy( dst.rowPointer( y ), src.rowPointer( y ), src.width() * src.strideBytes() );
		}
	}
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
