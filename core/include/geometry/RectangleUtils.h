#pragma once

#include <common/Array1DView.h>
#include <vecmath/Rect2i.h>
#include <vecmath/Rect2f.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector4f.h>

class RectangleUtils
{
public:

    // Fit an image inside a rectangle maximizing the resulting area and centering
    // (coordinates are rounded down).
    static Rect2i bestFitKeepAR( const Vector2i& imageSize, const Rect2i& window );

    // Fit an image inside a rectangle maximizing the resulting area and centering. 
    static Rect2f bestFitKeepAR( const Vector2f& imageSize, const Rect2f& window );

    // Given a standard rectangle, return its non-standard version along the y-axis:
    // the output has the same minimum() and maximum(), but has negative height.
    // Calling it twice flips it back.
    static Rect2f flipStandardizationY( const Rect2f& rect );

	static void writeScreenAlignedTriangleStrip(
		Array1DView< Vector4f > positions,
		Array1DView< Vector2f > textureCoordinates,
		const Rect2f& positionRectangle = Rect2f( -1, -1, 2, 2 ),
		float z = 0.f, float w = 1.f,
		const Rect2f& textureRectangle = Rect2f( 0, 0, 1, 1 )
	);

	static void writeScreenAlignedTriangleStripPositions(
		Array1DView< Vector4f > positions,
		const Rect2f& rect = Rect2f( -1, -1, 2, 2 ),
		float z = 0.f, float w = 1.f
	);

	// For a DirectX style rectangle,
	// pass in Rect2f( 0, 1, 1, -1 )
	static void writeScreenAlignedTriangleStripTextureCoordinates(
		Array1DView< Vector2f > textureCoordinates,
		const Rect2f& rect = Rect2f( 0, 0, 1, 1 )
	);
};