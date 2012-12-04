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
	static bool loadBinary( FILE* fp, std::vector< T >& output );

	template< typename T >
	static bool saveBinary( const std::vector< T >& input, const char* filename );

	template< typename T >
	static bool saveBinary( const std::vector< T >& input, FILE* fp );

	static bool saveTXT( const std::vector< float >& array, const char* filename );
	static bool saveTXT( const std::vector< int >& array, const char* filename );

	static bool saveTXT( const std::vector< Vector2f >& array, const char* filename );
	static bool saveTXT( const std::vector< Vector3f >& array, const char* filename );
	static bool saveTXT( const std::vector< Vector4f >& array, const char* filename );

	static bool saveTXT( const Array2D< float >& array, const char* filename );
	static bool saveTXT( const Array3D< float >& array, const char* filename );

	static bool saveTXT( const Array3D< Vector2f >& array, const char* filename );

	static bool saveTXT( const Array2D< Vector4f >& array, const char* filename );
};

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
