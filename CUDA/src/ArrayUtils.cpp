#include "ArrayUtils.h"

bool ArrayUtils::saveTXT( const Array2D< float2 >& array, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	fprintf( fp, "Size: %d x %d\n", array.width(), array.height() );
	fprintf( fp, "Format: float2\n" );

	int w = array.width();
	int h = array.height();

	for( int y = 0; y < h; ++y )
	{
		for( int x = 0; x < w; ++x )
		{
			int index = y * w + x;
			float2 v = array( x, y );
			fprintf( fp, "[%d] (%d %d): %f, %f\n", index, x, y, v.x, v.y );
		}
	}
	fclose( fp );

	return true;
}

bool ArrayUtils::saveTXT( const Array2D< float4 >& array, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	fprintf( fp, "Size: %d x %d\n", array.width(), array.height() );
	fprintf( fp, "Format: float4\n" );

	int w = array.width();
	int h = array.height();

	for( int y = 0; y < h; ++y )
	{
		for( int x = 0; x < w; ++x )
		{
			int index = y * array.width() + x;
			float4 v = array( x, y );
			fprintf( fp, "[%d] (%d %d): %f, %f, %f, %f\n", index, x, y, v.x, v.y, v.z, v.w );
		}
	}
	fclose( fp );

	return true;
}
