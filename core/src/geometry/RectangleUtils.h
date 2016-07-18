#pragma once

#include <vector>

#include <common/Array1D.h>
#include <common/Array1DView.h>
#include <vecmath/Rect2i.h>
#include <vecmath/Rect2f.h>
#include <vecmath/Matrix4f.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector2i.h>
#include <vecmath/Vector4f.h>
#include <vecmath/Vector4i.h>

namespace libcgt { namespace core { namespace geometry { namespace rectangleutils {

// Fit an image of a given aspect ratio inside a rectangle,
// maximizing the resulting area and centering (coordinates are rounded down).
Rect2i bestFitKeepAR( float imageAspectRatio, const Rect2i& window );

// Fit an image inside a rectangle maximizing the resulting area and centering
// (coordinates are rounded down).
Rect2i bestFitKeepAR( const Vector2i& imageSize, const Rect2i& window );

// Fit an image inside a rectangle maximizing the resulting area and centering.
Rect2f bestFitKeepAR( float imageAspectRatio, const Rect2f& window );

// Fit an image inside a rectangle maximizing the resulting area and centering.
Rect2f bestFitKeepAR( const Vector2f& imageSize, const Rect2f& window );

// Given a point p and a rectangle r, returns the normalized [0,1] coordinates of
// p inside r. Output is in [0,1]^2 if p is inside r. This function is valid elsewhere.
// Equivalent to ( p - r.origin ) / r.size.
Vector2f normalizedCoordinatesWithinRectangle( const Vector2f& p,
    const Rect2f& r );

// Returns a Matrix4f that takes a point in the from rectangle and
// maps it to a point in the to rectangle with scale and translation only.
// from.origin --> to.origin
// from.limit --> to.limit
Matrix4f transformBetween( const Rect2f& from, const Rect2f& to );

// Translate (moves) r: r.origin += delta.
Rect2f translate( const Rect2f& r, const Vector2f& delta );

// Translate (moves) r: r.origin += delta.
Rect2i translate( const Rect2i& r, const Vector2i& delta );

// [Requires a standard rectangle].
// Returns the exact same rectangle as r, but in a coordinate system
// that counts from [0, width) but where x points left.
// This function is useful for handling rectangles within 2D images where
// the x axis points left.
//
// Let r = { 1, y, 2, height }.
// Then the bottom right corner has x = 1 + 2 = 3.
// If width = 5, returns { 2, y, 2, height }: the new origin.x = 2.
Rect2f flipX( const Rect2f& r, float width );

// [Requires a standard rectangle].
// Returns the exact same rectangle as r, but in a coordinate system
// that counts from [0, width) but where x points left.
// This function is useful for handling rectangles within 2D images where
// the x axis points left.
//
// Let r = { 1, y, 2, height }.
// Then the bottom right corner has x = 1 + 2 = 3.
// If width = 5, returns { 2, y, 2, height }: the new origin.x = 2.
Rect2i flipX( const Rect2i& r, int width );

// [Requires a standard rectangle].
// Returns the exact same rectangle as this, but in a coordinate system
// that counts from [0, height) but where y points down.
// This function is useful for handling rectangles within 2D images where
// the y axis points down.
//
// Let r = { x, 1, width, 2 }.
// Then the top left corner has y = 1 + 2 = 3.
// If height = 5, returns { x, 2, width, 2 }: the new origin.y = 2.
Rect2f flipY( const Rect2f& r, float height );

// [Requires a standard rectangle].
// Returns the exact same rectangle as this, but in a coordinate system
// that counts from [0, height) but where y points down.
// This function is useful for handling rectangles within 2D images where
// the y axis points down.
//
// Let r = { x, 1, width, 2 }.
// Then the top left corner has y = 1 + 2 = 3.
// If height = 5, returns { x, 2, width, 2 }: the new origin.y = 2.
Rect2i flipY( const Rect2i& r, int height );

// Return the non-standard version of rect along the x-axis.
// The output has the same minimum() and maximum(), but has negative height.
// Calling it twice flips it back.
Rect2f flipStandardizationX( const Rect2f& rect );

// Return the non-standard version of rect along the y-axis.
// the output has the same minimum() and maximum(), but has negative height.
// Calling it twice flips it back.
Rect2f flipStandardizationY( const Rect2f& rect );

// Shrink a rectangle by delta on all four sides.
Rect2i inset( const Rect2i& r, int delta );

// Shrink a rectangle by xy.x from both left and right, and xy.y from both
// bottom and top.
Rect2i inset( const Rect2i& r, const Vector2i& xy );

void writeScreenAlignedTriangleStrip(
    Array1DView< Vector4f > positions,
    Array1DView< Vector2f > textureCoordinates,
    const Rect2f& positionRectangle = Rect2f{ { -1, -1 }, { 2, 2 } },
    float z = 0.f, float w = 1.f,
    const Rect2f& textureRectangle = Rect2f{ { 1, 1 } }
);

// Write the positions of a screen-aligned rectangle as a triangle strip
// into positions.
//
// The default rect is already in clip space, and will not need a projection
// matrix.
//
// If rect = { x, y, width, height },
// then the projection matrix should be:
// orthographicProjection( 0, 0, width, height ).
void writeScreenAlignedTriangleStripPositions(
    Array1DView< Vector4f > positions,
    const Rect2f& rect = Rect2f{ { -1, -1 }, { 2, 2 } },
    float z = 0.f, float w = 1.f
);

// For a DirectX style rectangle,
// pass in Rect2f( 0, 1, 1, -1 )
void writeScreenAlignedTriangleStripTextureCoordinates(
    Array1DView< Vector2f > textureCoordinates,
    const Rect2f& rect = Rect2f{ { 1, 1 } }
);

// Returns a standard rectangle (square) with the given center and side length.
Rect2f makeSquare( const Vector2f& center, float sideLength );

// [Requires a standard rectangle].
// Returns the 4 corners in counterclockwise order (if y points up).
Array1D< Vector4f > corners( const Rect2f& r, float z = 0.0f, float w = 1.0f );

// [Requires a standard rectangle].
// Returns the 4 corners in counterclockwise order (if y points up).
Array1D< Vector4i > corners( const Rect2i& r, int z = 0, int w = 1 );

// Returns the 6 indices in corners() of a solid box with triangle strip
// topology.
Array1D< int > solidTriangleListIndices();

// Returns the 4 indices in corners() of a solid box with triangle list
// topology.
Array1D< int > solidTriangleStripIndices();

// Returns the 8 indices in corners() of a solid box with line list topology.
Array1D< int > wireframeLineListIndices();

} } } } // rectangleutils, geometry, core, libcgt
