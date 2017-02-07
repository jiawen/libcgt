#include "TexturedRectangle.h"

#include <cassert>

#include "libcgt/core/common/ArrayUtils.h"
#include "libcgt/core/geometry/RectangleUtils.h"

using libcgt::core::geometry::writeScreenAlignedTriangleStripPositions;
using libcgt::core::geometry::writeScreenAlignedTriangleStripTextureCoordinates;

TexturedRectangle::TexturedRectangle( const Rect2f& positionRect,
    const Rect2f& texCoordsRect ) :
    GLDrawable( GLPrimitiveType::TRIANGLE_STRIP, calculator() )
{
    updatePositions( positionRect );
    updateTexCoords( texCoordsRect );
}

void TexturedRectangle::updatePositions( const Rect2f& positionRect )
{
    auto mb = mapAttribute< Vector4f >( 0 );
    writeScreenAlignedTriangleStripPositions( mb.view(), positionRect );
}

void TexturedRectangle::updatePositions(
    const std::vector< Vector3f >& positions )
{
    assert( positions.size() >= 4 );

    auto mb = mapAttribute< Vector4f >( 0 );
    Array1DWriteView< Vector4f > dst = mb.view();
    dst[ 0 ] = { positions[ 0 ], 1.0f };
    dst[ 1 ] = { positions[ 1 ], 1.0f };
    dst[ 2 ] = { positions[ 2 ], 1.0f };
    dst[ 3 ] = { positions[ 3 ], 1.0f };
}

void TexturedRectangle::updatePositions(
    const std::vector< Vector4f >& positions )
{
    auto mb = mapAttribute< Vector4f >( 0 );
    Array1DWriteView< Vector4f > dst = mb.view();
    dst[ 0 ] = positions[ 0 ];
    dst[ 1 ] = positions[ 1 ];
    dst[ 2 ] = positions[ 2 ];
    dst[ 3 ] = positions[ 3 ];
}

void TexturedRectangle::updateTexCoords( const Rect2f& texCoordsRect )
{
    auto mb = mapAttribute< Vector2f >( 1 );
    writeScreenAlignedTriangleStripTextureCoordinates(
        mb.view(), texCoordsRect );
}

// static
PlanarVertexBufferCalculator TexturedRectangle::calculator()
{
    const int NUM_VERTICES = 4;
    PlanarVertexBufferCalculator calculator( NUM_VERTICES );
    calculator.addAttribute< Vector4f >();
    calculator.addAttribute< Vector2f >();
    return calculator;
}
