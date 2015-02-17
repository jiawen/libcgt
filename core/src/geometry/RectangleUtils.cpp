#include "geometry/RectangleUtils.h"

#include <algorithm>
#include <cassert>

// static
Rect2i RectangleUtils::bestFitKeepAR( float imageAspectRatio, const Rect2i& window )
{
    int w = std::min( window.width(), static_cast< int >( window.height() * imageAspectRatio ) );
    int h = std::min( static_cast< int >( window.width() / imageAspectRatio ), window.height() );
    
    int x = window.origin().x + ( window.width() - w ) / 2;
    int y = window.origin().y + ( window.height() - h ) / 2;
    
    return{ x, y, w, h };
}

// static
Rect2i RectangleUtils::bestFitKeepAR( const Vector2i& imageSize, const Rect2i& window )
{
    return bestFitKeepAR( static_cast< float >( imageSize.x ) / imageSize.y, window );
}

// static
Rect2f RectangleUtils::bestFitKeepAR( float imageAspectRatio, const Rect2f& window )
{
    float w = std::min( window.width(), window.height() * imageAspectRatio );
    float h = std::min( window.width() / imageAspectRatio, window.height() );
    
    float x = window.origin().x + 0.5f * ( window.width() - w );
    float y = window.origin().y + 0.5f * ( window.height() - h );
    
    return{ x, y, w, h };
}

// static
Rect2f RectangleUtils::bestFitKeepAR( const Vector2f& imageSize, const Rect2f& window )
{
    return bestFitKeepAR( imageSize.x / imageSize.y, window );
}

// static
Matrix4f RectangleUtils::transformBetween( const Rect2f& from, const Rect2f& to )
{
    // Given a point p:
    // 0. p0 = p - from.origin
    // 1. p1 = p0 / from.size
    // 2. p2 = p1 * to.size
    // 3. p3 = p2 + to.origin

    return Matrix4f::translation( to.origin().x, to.origin().y, 0 ) *
        Matrix4f::scaling( to.size().x, to.size().y, 1 ) *
        Matrix4f::scaling( 1.0f / from.size().x, 1.0f / from.size().y, 1 ) *
        Matrix4f::translation( -from.origin().x, -from.origin().y, 0 );
}

// static
Vector2f RectangleUtils::normalizedCoordinatesWithinRectangle( const Vector2f& p,
    const Rect2f& r )
{
    return ( p - r.origin() ) / r.size();
}

// static
Rect2f RectangleUtils::flipX( const Rect2f& r, float width )
{
    assert( r.isStandardized() );
	Vector2f origin;
	origin.x = width - r.limit().x;
	origin.y = r.origin().y;

    return{ origin, r.size() };
}

// static
Rect2i RectangleUtils::flipX( const Rect2i& r, int width )
{
    assert( r.isStandardized() );
	Vector2i origin;
	origin.x = width - r.limit().x;
	origin.y = r.origin().y;

    return{ origin, r.size() };
}

// static
Rect2f RectangleUtils::flipY( const Rect2f& r, float height )
{
    assert( r.isStandardized() );
	Vector2f origin;
	origin.x = r.origin().x;
	origin.y = height - r.limit().y;

    return{ origin, r.size() };
}

// static
Rect2i RectangleUtils::flipY( const Rect2i& r, int height )
{
    assert( r.isStandardized() );
	Vector2i origin;
	origin.x = r.origin().x;
	origin.y = height - r.limit().y;

    return{ origin, r.size() };
}

// static
Rect2f RectangleUtils::flipStandardizationX( const Rect2f& rect )
{
    Rect2f output = rect;
    output.origin().x = output.origin().x + output.size().x;
    output.size().x = -output.size().x;
    return output;
}

// static
Rect2f RectangleUtils::flipStandardizationY( const Rect2f& rect )
{
    Rect2f output = rect;
    output.origin().y = output.origin().y + output.size().y;
    output.size().y = -output.size().y;
    return output;
}

// static
void RectangleUtils::writeScreenAlignedTriangleStrip(
		Array1DView< Vector4f > positions,
		Array1DView< Vector2f > textureCoordinates,
		const Rect2f& positionRectangle,
		float z, float w,
		const Rect2f& textureRectangle )
{
	writeScreenAlignedTriangleStripPositions(
		positions, positionRectangle, z, w );
	writeScreenAlignedTriangleStripTextureCoordinates(
		textureCoordinates, textureRectangle );
}

// static
void RectangleUtils::writeScreenAlignedTriangleStripPositions(
		Array1DView< Vector4f > positions,
		const Rect2f& rect,
		float z, float w )
{
	positions[ 0 ] = Vector4f( rect.origin().x, rect.origin().y, z, w );
	positions[ 1 ] = Vector4f( rect.limit().x, rect.origin().y, z, w );
	positions[ 2 ] = Vector4f( rect.origin().x, rect.limit().y, z, w );
	positions[ 3 ] = Vector4f( rect.limit().x, rect.limit().y, z, w );
}

// 	static
void RectangleUtils::writeScreenAlignedTriangleStripTextureCoordinates(
		Array1DView< Vector2f > textureCoordinates,
		const Rect2f& rect )
{
    textureCoordinates[ 0 ] = Vector2f{ rect.origin().x, rect.origin().y };
    textureCoordinates[ 1 ] = Vector2f{ rect.limit().x, rect.origin().y };
    textureCoordinates[ 2 ] = Vector2f{ rect.origin().x, rect.limit().y };
    textureCoordinates[ 3 ] = Vector2f{ rect.limit().x, rect.limit().y };
}

// static
Rect2f RectangleUtils::makeSquare( const Vector2f& center, float sideLength )
{
    return Rect2f( center - 0.5f * Vector2f( sideLength ), Vector2f( sideLength ) );
}

// static
std::vector< Vector2f > RectangleUtils::corners( const Rect2f& r )
{
    assert( r.isStandardized() );
    return
    {
        r.origin(),
        Vector2f{ r.limit().x, r.origin().y },
        r.limit(),
        Vector2f{ r.origin().x, r.limit().y }
    };
}

// static
std::vector< Vector2i > RectangleUtils::corners( const Rect2i& r )
{
    assert( r.isStandardized() );
    return
    {
        r.origin(),
        Vector2i{ r.limit().x, r.origin().y },
        r.limit(),
        Vector2i{ r.origin().x, r.limit().y }
    };
}
