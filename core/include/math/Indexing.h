#pragma once

#include <vecmath/Vector2i.h>
#include <vecmath/Vector3i.h>

// Fun indexing math (1D <--> ND coordinates)
class Indexing
{
public:

	// unwraps a 1D linear index into a 2D subscript
	static void indexToSubscript( int index, int width, int& x, int& y );
	static Vector2i indexToSubscript( int index, int width );

	// unwraps a 1D linear index into a 3D subscript
	static void indexToSubscript( int index, int width, int height, int& x, int&y, int& z );
	static Vector3i indexToSubscript( int index, int width, int height );

	// wraps a 2D subscript into a 1D linear index
	static int subscriptToIndex( int x, int y, int width );
	static int subscriptToIndex( const Vector2i& xy, int width );
	static int subscriptToIndex( const Vector2i& xy, const Vector2i& size ); // size.y is ignored

	// wraps a 3D subscript into a 1D linear index
	static int subscriptToIndex( int x, int y, int z, int width, int height );
	static int subscriptToIndex( const Vector3i& xy, int width, int height );
	static int subscriptToIndex( const Vector3i& xy, const Vector3i& size ); // size.z is ignored

};
