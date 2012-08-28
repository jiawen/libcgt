#pragma once

#include <vector>
#include "Array2D.h"
#include "Array3D.h"

#include <vecmath/Vector2f.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>

class ArrayUtils
{
public:

	template< typename T >
	static bool saveBinary( const std::vector< T >& input, const char* filename );

	static bool saveTXT( const std::vector< float >& array, const char* filename );
	static bool saveTXT( const Array2D< float >& array, const char* filename );
	static bool saveTXT( const Array3D< float >& array, const char* filename );

	static bool saveTXT( const Array3D< Vector2f >& array, const char* filename );

	static bool saveTXT( const Array2D< Vector4f >& array, const char* filename );
};

template< typename T >
// static
bool ArrayUtils::saveBinary( const std::vector< T >& input, const char* filename )
{
	FILE* fp = fopen( filename, "wb" );
	if( fp == nullptr )
	{
		return false;
	}

	int length = static_cast< int >( input.size() );
	fwrite( &length, sizeof( int ), 1, fp );

	fwrite( input.data(), sizeof( T ), size, fp );

	fclose( fp );

	return true;
}