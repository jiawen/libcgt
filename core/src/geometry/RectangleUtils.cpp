#include "geometry/RectangleUtils.h"

#include <algorithm>
#include <cassert>

namespace libcgt { namespace core { namespace geometry {

Rect2i bestFitKeepAR( float imageAspectRatio, const Rect2i& window )
{
    int w = std::min( window.width(),
        static_cast< int >( window.height() * imageAspectRatio ) );
    int h = std::min( static_cast< int >( window.width() / imageAspectRatio ),
        window.height() );

    int x = window.left() + ( window.width() - w ) / 2;
    int y = window.bottom() + ( window.height() - h ) / 2;

    return{ { x, y }, { w, h } };
}

Rect2i bestFitKeepAR( const Vector2i& imageSize, const Rect2i& window )
{
    return bestFitKeepAR(
        static_cast< float >( imageSize.x ) / imageSize.y, window );
}

Rect2f bestFitKeepAR( float imageAspectRatio, const Rect2f& window )
{
    float w = std::min( window.width(), window.height() * imageAspectRatio );
    float h = std::min( window.width() / imageAspectRatio, window.height() );

    float x = window.left() + 0.5f * ( window.width() - w );
    float y = window.bottom() + 0.5f * ( window.height() - h );

    return{ { x, y }, { w, h } };
}

Rect2f bestFitKeepAR( const Vector2f& imageSize, const Rect2f& window )
{
    return bestFitKeepAR( imageSize.x / imageSize.y, window );
}

Matrix4f transformBetween( const Rect2f& from, const Rect2f& to )
{
    // Given a point p:
    // 0. p0 = p - from.origin
    // 1. p1 = p0 / from.size
    // 2. p2 = p1 * to.size
    // 3. p3 = p2 + to.origin

    return Matrix4f::translation( { to.origin.x, to.origin.y, 0.0f } ) *
        Matrix4f::scaling( { to.size.x, to.size.y, 1.0f } ) *
        Matrix4f::scaling( { 1.0f / from.size.x, 1.0f / from.size.y, 1.0f } ) *
        Matrix4f::translation( { -from.origin.x, -from.origin.y, 0.0f } );
}

Vector2f normalizedCoordinatesWithinRectangle( const Vector2f& p,
    const Rect2f& r )
{
    return ( p - r.origin ) / r.size;
}

Rect2f translate( const Rect2f& r, const Vector2f& delta )
{
    Rect2f r2 = r;
    r2.origin += delta;
    return r2;
}

Rect2i translate( const Rect2i& r, const Vector2i& delta )
{
    Rect2i r2 = r;
    r2.origin += delta;
    return r2;
}

Rect2f flipX( const Rect2f& r, float width )
{
    assert( r.isStandard() );

    Vector2f origin;
    origin.x = width - r.right();
    origin.y = r.origin.y;

    return{ origin, r.size };
}

Rect2i flipX( const Rect2i& r, int width )
{
    assert( r.isStandard() );

    Vector2i origin;
    origin.x = width - r.right();
    origin.y = r.origin.y;

    return{ origin, r.size };
}

Rect2f flipY( const Rect2f& r, float height )
{
    assert( r.isStandard() );

    Vector2f origin;
    origin.x = r.origin.x;
    origin.y = height - r.top();

    return{ origin, r.size };
}

Rect2i flipY( const Rect2i& r, int height )
{
    assert( r.isStandard() );

    Vector2i origin;
    origin.x = r.origin.x;
    origin.y = height - r.top();

    return{ origin, r.size };
}

Rect2f flipStandardizationX( const Rect2f& rect )
{
    Rect2f output = rect;
    output.origin.x = output.origin.x + output.size.x;
    output.size.x = -output.size.x;
    return output;
}

Rect2f flipStandardizationY( const Rect2f& rect )
{
    Rect2f output = rect;
    output.origin.y = output.origin.y + output.size.y;
    output.size.y = -output.size.y;
    return output;
}

Rect2i inset( const Rect2i& r, int delta )
{
    return inset( r, { delta, delta } );
}

Rect2i inset( const Rect2i& r, const Vector2i& xy )
{
    return
    {
        r.origin + xy,
        r.size - 2 * xy
    };
}

void writeScreenAlignedTriangleStrip(
    Array1DWriteView< Vector4f > positions,
    Array1DWriteView< Vector2f > textureCoordinates,
    const Rect2f& positionRectangle,
    float z, float w,
    const Rect2f& textureRectangle )
{
    writeScreenAlignedTriangleStripPositions(
        positions, positionRectangle, z, w );
    writeScreenAlignedTriangleStripTextureCoordinates(
        textureCoordinates, textureRectangle );
}

void writeScreenAlignedTriangleStripPositions(
    Array1DWriteView< Vector4f > positions,
    const Rect2f& rect,
    float z, float w )
{
    Array1D< Vector4f > pos = corners( rect, z, w );
    Array1D< int > idx = solidRectTriangleStripIndices();

    for( int i = 0; i < idx.size(); ++i )
    {
        positions[ i ] = pos[ idx[ i ] ];
    }
}

//  static
void writeScreenAlignedTriangleStripTextureCoordinates(
    Array1DWriteView< Vector2f > textureCoordinates,
    const Rect2f& rect )
{
    textureCoordinates[ 0 ] = rect.leftBottom();
    textureCoordinates[ 1 ] = rect.rightBottom();
    textureCoordinates[ 2 ] = rect.leftTop();
    textureCoordinates[ 3 ] = rect.rightTop();
}

Rect2f makeRect( const Vector2f& center, float sideLength )
{
    return makeRect( center, Vector2f{ sideLength } );
}

Rect2f makeRect( const Vector2f& center, const Vector2f& sideLengths )
{
    return
    {
        center - 0.5f * sideLengths,
        sideLengths
    };
}

Array1D< Vector4f > corners( const Rect2f& r, float z, float w )
{
    assert( r.isStandard() );

    return
    {
        Vector4f{ r.leftBottom(), z, w },
        Vector4f{ r.rightBottom(), z, w },
        Vector4f{ r.rightTop(), z, w },
        Vector4f{ r.leftTop(), z, w }
    };
}

Array1D< Vector4i > corners( const Rect2i& r, int z, int w )
{
    assert( r.isStandard() );

    return
    {
        Vector4i{ r.leftBottom(), z, w },
        Vector4i{ r.rightBottom(), z, w },
        Vector4i{ r.rightTop(), z, w },
        Vector4i{ r.leftTop(), z, w }
    };
}

Array1D< int > solidRectTriangleListIndices()
{
    return{ 0, 1, 3, 3, 1, 2 };
}

Array1D< int > solidRectTriangleStripIndices()
{
    return{ 0, 1, 3, 2 };
}

Array1D< int > wireframeRectLineListIndices()
{
    return{ 0, 1, 1, 2, 2, 3, 3, 1 };
}

} } } // geometry, core, libcgt
