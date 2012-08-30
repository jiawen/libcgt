#include "ArrayUtils.h"

#include <common/ArrayUtils.h>

// static
bool libcgt::cuda::ArrayUtils::saveTXT( const Array2D< float2 >& array, const char* filename )
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

// static
bool libcgt::cuda::ArrayUtils::saveTXT( const Array2D< float4 >& array, const char* filename )
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

// static
bool libcgt::cuda::ArrayUtils::saveTXT( const std::vector< int3 >& array, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == nullptr )
	{
		return false;
	}

	fprintf( fp, "Size: %d\n", array.size() );
	fprintf( fp, "Format: int3\n" );

	int length = static_cast< int >( array.size() );
	for( int i = 0; i < length; ++i )
	{
		int3 v = array[i];
		fprintf( fp, "[%d]: %d, %d, %d\n", i, v.x, v.y, v.z );
	}

	fclose( fp );

	return true;
}

// static
bool libcgt::cuda::ArrayUtils::saveTXT( const DeviceArray2D< float >& array, const char* filename )
{
	Array2D< float > h_array( array.width(), array.height() );
	array.copyToHost( h_array );
	return ::ArrayUtils::saveTXT( h_array, filename );
}

// static
bool libcgt::cuda::ArrayUtils::saveTXT( const DeviceArray2D< float4 >& array, const char* filename )
{
	Array2D< float4 > h_array( array.width(), array.height() );
	array.copyToHost( h_array );
	return saveTXT( h_array, filename );
}

// static
bool libcgt::cuda::ArrayUtils::saveTXT( const DeviceVector< int3 >& array, const char* filename )
{
	std::vector< int3 > h_array( array.length() );
	array.copyToHost( h_array );
	return saveTXT( h_array, filename );
}
