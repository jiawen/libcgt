#include "SolidBox.h"

#include <common/ArrayUtils.h>
#include <geometry/BoxUtils.h>
#include <PlanarVertexBufferCalculator.h>
#include <GLPrimitiveType.h>

using libcgt::core::arrayutils::fill;
using libcgt::core::geometry::boxutils::writeTriangleListPositions;

SolidBox::SolidBox( const Box3f& box, const Vector4f& color ) :
    GLDrawable( GLPrimitiveType::TRIANGLES, calculator() )
{
    updatePositions( box );
    updateColors( color );
}

void SolidBox::updatePositions( const Box3f& box,
    const Matrix4f& worldFromBox )
{
    auto mb = mapAttribute< Vector4f >( 0 );
    writeTriangleListPositions( box, worldFromBox, mb.view() );
}

void SolidBox::updateColors( const Vector4f& color )
{
    auto mb = mapAttribute< Vector4f >( 1 );
    fill( mb.view(), color );
}

// static
PlanarVertexBufferCalculator SolidBox::calculator()
{
    const int NUM_FACES = 6;
    const int NUM_VERTICES_PER_FACE = 6;
    const int NUM_VERTICES = NUM_FACES * NUM_VERTICES_PER_FACE;
    PlanarVertexBufferCalculator calculator( NUM_VERTICES );
    calculator.addAttribute( 4, sizeof( float ) );
    calculator.addAttribute( 4, sizeof( float ) );
    return calculator;
}
