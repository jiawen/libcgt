#include "common/ArrayUtils.h"

#include <cstdlib>

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
bool ArrayUtils::saveTXT( const std::vector< float >& array, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	int retVal;
	int n = static_cast< int >( array.size() );

	retVal = fprintf( fp, "Size: %d\n", n );
	if( retVal < 0 )
	{
		return false;
	}

	for( int i = 0; i < n; ++i )
	{
		fprintf( fp, "[%d]: %f\n", i, array[ i ] );
	}
	
	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( const Array2D< float >& array, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}
	
	int retVal;
	
	retVal = fprintf( fp, "Size: %d x %d\n", array.width(), array.height() );
	if( retVal < 0 )
	{
		return false;
	}

	retVal = fprintf( fp, "Format: float\n" );
	if( retVal < 0 )
	{
		return false;
	}

	int w = array.width();
	int h = array.height();

	for( int y = 0; y < h; ++y )
	{
		for( int x = 0; x < w; ++x )
		{
			int index = y * w + x;
			float v = array( x, y );
			fprintf( fp, "[%d] (%d %d): %f\n", index, x, y, v );
		}
	}
	
	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( const Array3D< float >& array, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	int retVal;

	retVal = fprintf( fp, "Size: %d x %d x %d\n", array.width(), array.height(), array.depth() );
	if( retVal < 0 )
	{
		return false;
	}

	retVal = fprintf( fp, "Format: float\n" );
	if( retVal < 0 )
	{
		return false;
	}

	int w = array.width();
	int h = array.height();
	int d = array.depth();

	for( int z = 0; z < d; ++z )
	{
		for( int y = 0; y < h; ++y )
		{
			for( int x = 0; x < w; ++x )
			{
				int index = z * w * h + y * w + x;
				float v = array( x, y, z );
				fprintf( fp, "[%d] (%d %d %d): %f\n", index, x, y, z, v );
			}
		}
	}
	
	retVal = fclose( fp );
	return( retVal == 0 );
}
