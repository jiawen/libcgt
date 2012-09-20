#include "math/Indexing.h"

// static
void Indexing::indexToSubscript2D( int index, int width, int& x, int& y )
{
	y = index / width;
	x = index - y * width;	
}

// static
Vector2i Indexing::indexToSubscript2D( int index, int width )
{
	int x;
	int y;
	indexToSubscript2D( index, width, x, y );
	return Vector2i( x, y );
}

// static
void Indexing::indexToSubscript3D( int index, int width, int height, int& x, int& y, int& z )
{
	int wh = width * height;
	z = index / wh;

	int ky = index - z * wh;
	y = ky / width;

	x = ky - y * width;
}

// static
Vector3i Indexing::indexToSubscript3D( int index, int width, int height )
{
	int x;
	int y;
	int z;
	indexToSubscript3D( index, width, height, x, y, z );
	return Vector3i( x, y, z );
}

// static
int Indexing::subscript2DToIndex( int x, int y, int width )
{
	return( y * width + x );
}

// static
int Indexing::subscript2DToIndex( const Vector2i& xy, int width )
{
	return subscript2DToIndex( xy.x, xy.y, width );
}

// static
int Indexing::subscript2DToIndex( const Vector2i& xy, const Vector2i& size )
{
	return subscript2DToIndex( xy.x, xy.y, size.x );
}

// static
int Indexing::subscript3DToIndex( int x, int y, int z, int width, int height )
{
	return( z * width * height + y * width + x );
}

// static
int Indexing::subscript3DToIndex( const Vector3i& xy, int width, int height )
{
	return subscript3DToIndex( xy.x, xy.y, xy.z, width, height );
}

// static
int Indexing::subscript3DToIndex( const Vector3i& xy, const Vector3i& size )
{
	return subscript3DToIndex( xy.x, xy.y, xy.z, size.x, size.y );
}
