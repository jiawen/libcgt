#include "math/Indexing.h"

// static
void Indexing::indexToSubscript( int index, int width, int& x, int& y )
{
	y = index / width;
	x = index - y * width;	
}

// static
Vector2i Indexing::indexToSubscript( int index, int width )
{
	int x;
	int y;
	indexToSubscript( index, width, x, y );
	return Vector2i( x, y );
}

// static
void Indexing::indexToSubscript( int index, int width, int height, int& x, int& y, int& z )
{
	int wh = width * height;
	z = index / wh;

	int ky = index - z * wh;
	y = ky / width;

	x = ky - y * width;
}

// static
Vector3i Indexing::indexToSubscript( int index, int width, int height )
{
	int x;
	int y;
	int z;
	indexToSubscript( index, width, height, x, y, z );
	return Vector3i( x, y, z );
}

// static
int Indexing::subscriptToIndex( int x, int y, int width )
{
	return( y * width + x );
}

// static
int Indexing::subscriptToIndex( const Vector2i& xy, int width )
{
	return subscriptToIndex( xy.x, xy.y, width );
}

// static
int Indexing::subscriptToIndex( const Vector2i& xy, const Vector2i& size )
{
	return subscriptToIndex( xy.x, xy.y, size.x );
}

// static
int Indexing::subscriptToIndex( int x, int y, int z, int width, int height )
{
	return( z * width * height + y * width + x );
}

// static
int Indexing::subscriptToIndex( const Vector3i& xy, int width, int height )
{
	return subscriptToIndex( xy.x, xy.y, xy.z, width, height );
}

// static
int Indexing::subscriptToIndex( const Vector3i& xy, const Vector3i& size )
{
	return subscriptToIndex( xy.x, xy.y, xy.z, size.x, size.y );
}
