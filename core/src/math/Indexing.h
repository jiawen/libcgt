#pragma once

#include <vecmath/Vector2i.h>
#include <vecmath/Vector3i.h>

// Indexing math: (1D <--> ND coordinates).
namespace libcgt { namespace core {

// Unwrap a 1D linear index into a 2D subscript.
void indexToSubscript2D( int index, int width, int& x, int& y );

// Unwrap a 1D linear index into a 2D subscript.
Vector2i indexToSubscript2D( int index, int width );

// Unwrap a 1D linear index into a 3D subscript.
void indexToSubscript3D( int index, int width, int height,
    int& x, int&y, int& z );

// Unwrap a 1D linear index into a 3D subscript.
Vector3i indexToSubscript3D( int index, int width, int height );

// wraps a 2D subscript into a 1D linear index
int subscript2DToIndex( int x, int y, int width );
int subscript2DToIndex( const Vector2i& xy, int width );
int subscript2DToIndex( const Vector2i& xy, const Vector2i& size ); // size.y is ignored

// wraps a 3D subscript into a 1D linear index
int subscript3DToIndex( int x, int y, int z, int width, int height );
int subscript3DToIndex( const Vector3i& xy, int width, int height );
int subscript3DToIndex( const Vector3i& xy, const Vector3i& size ); // size.z is ignored

} } // core, libcgt

#include "Indexing.inl"
