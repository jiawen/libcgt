#pragma once

#include <vector>

#include <common/Array1DView.h>
#include <vecmath/Rect2i.h>
#include <vecmath/Rect2f.h>
#include <vecmath/Matrix4f.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector4f.h>

class RectangleUtils
{
public:

    // Fit an image of a given aspect ratio inside a rectangle,
    // maximizing the resulting area and centering (coordinates are rounded down).
    static Rect2i bestFitKeepAR( float imageAspectRatio, const Rect2i& window );

    // Fit an image inside a rectangle maximizing the resulting area and centering
    // (coordinates are rounded down).
    static Rect2i bestFitKeepAR( const Vector2i& imageSize, const Rect2i& window );

    // Fit an image inside a rectangle maximizing the resulting area and centering. 
    static Rect2f bestFitKeepAR( float imageAspectRatio, const Rect2f& window );

    // Fit an image inside a rectangle maximizing the resulting area and centering. 
    static Rect2f bestFitKeepAR( const Vector2f& imageSize, const Rect2f& window );

    // Given a point p and a rectangle r, returns the normalized [0,1] coordinates of
    // p inside r. Output is in [0,1]^2 if p is inside r. This function is valid elsewhere.
    // Equivalent to ( p - r.origin ) / r.size.
    static Vector2f normalizedCoordinatesWithinRectangle( const Vector2f& p,
        const Rect2f& r );

    // Returns a Matrix4f that takes a point in the from rectangle and
    // maps it to a point in the to rectangle with scale and translation only.
    // from.origin --> to.origin
    // from.limit --> to.limit
    static Matrix4f transformBetween( const Rect2f& from, const Rect2f& to );

    // [Requires a standard rectangle].
    // Returns the exact same rectangle as r, but in a coordinate system
    // that counts from [0, width) but where x points left.
	// This function is useful for handling rectangles within 2D images where
    // the x axis points left.
    //
    // Let r = { 1, y, 2, height }.
    // Then the bottom right corner has x = 1 + 2 = 3.
    // If width = 5, returns { 2, y, 2, height }: the new origin.x = 2.
    static Rect2f flipX( const Rect2f& r, float width );

    // [Requires a standard rectangle].
    // Returns the exact same rectangle as r, but in a coordinate system
    // that counts from [0, width) but where x points left.
	// This function is useful for handling rectangles within 2D images where
    // the x axis points left.
    //
    // Let r = { 1, y, 2, height }.
    // Then the bottom right corner has x = 1 + 2 = 3.
    // If width = 5, returns { 2, y, 2, height }: the new origin.x = 2.
    static Rect2i flipX( const Rect2i& r, int width );

    // [Requires a standard rectangle].
    // Returns the exact same rectangle as this, but in a coordinate system
    // that counts from [0, height) but where y points down.
	// This function is useful for handling rectangles within 2D images where
    // the y axis points down.
    //
    // Let r = { x, 1, width, 2 }.
    // Then the top left corner has y = 1 + 2 = 3.
    // If height = 5, returns { x, 2, width, 2 }: the new origin.y = 2.
    static Rect2f flipY( const Rect2f& r, float height );

    // [Requires a standard rectangle].
    // Returns the exact same rectangle as this, but in a coordinate system
    // that counts from [0, height) but where y points down.
	// This function is useful for handling rectangles within 2D images where
    // the y axis points down.
    //
    // Let r = { x, 1, width, 2 }.
    // Then the top left corner has y = 1 + 2 = 3.
    // If height = 5, returns { x, 2, width, 2 }: the new origin.y = 2.
    static Rect2i flipY( const Rect2i& r, int height );

    // Return the non-standard version of rect along the x-axis.
    // The output has the same minimum() and maximum(), but has negative height.
    // Calling it twice flips it back.
    static Rect2f flipStandardizationX( const Rect2f& rect );

    // Return the non-standard version of rect along the y-axis.
    // the output has the same minimum() and maximum(), but has negative height.
    // Calling it twice flips it back.
    static Rect2f flipStandardizationY( const Rect2f& rect );
    
	static void writeScreenAlignedTriangleStrip(
		Array1DView< Vector4f > positions,
		Array1DView< Vector2f > textureCoordinates,
        const Rect2f& positionRectangle = Rect2f{ -1, -1, 2, 2 },
		float z = 0.f, float w = 1.f,
        const Rect2f& textureRectangle = Rect2f{ 0, 0, 1, 1 }
	);

    // Write the positions of a screen-aligned rectangle as a triangle strip
    // into positions.
    //
    // The default rect is already in clip space, and 
    // projection matrix is needed.
    //
    // If rect = { x, y, width, height },
    // then the projection matrix should be:
    // orthographicProjection( 0, 0, width, height ).
	static void writeScreenAlignedTriangleStripPositions(
		Array1DView< Vector4f > positions,
        const Rect2f& rect = Rect2f{ -1, -1, 2, 2 },
		float z = 0.f, float w = 1.f
	);

	// For a DirectX style rectangle,
	// pass in Rect2f( 0, 1, 1, -1 )
	static void writeScreenAlignedTriangleStripTextureCoordinates(
		Array1DView< Vector2f > textureCoordinates,
        const Rect2f& rect = Rect2f{ 0, 0, 1, 1 }
	);

    // Returns a standard rectangle (square) with the given center and side length.
    static Rect2f makeSquare( const Vector2f& center, float sideLength );

    // [Requires a standard rectangle].
    // Returns the 4 corners in counterclockwise order (if y points up).
    static std::vector< Vector2f > corners( const Rect2f& r );

    // [Requires a standard rectangle].
    // Returns the 4 corners in counterclockwise order (if y points up).
    static std::vector< Vector2i > corners( const Rect2i& r );
};
