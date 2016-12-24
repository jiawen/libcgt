#include "SolidBox.h"

#include "libcgt/core/common/ArrayUtils.h"
#include "libcgt/core/geometry/BoxUtils.h"
#include "libcgt/GL/PlanarVertexBufferCalculator.h"
#include "libcgt/GL/GLPrimitiveType.h"

using libcgt::core::arrayutils::fill;
using libcgt::core::geometry::writeTriangleListPositions;

SolidBox::SolidBox( const Box3f& box, const Matrix4f& worldFromBox,
    const Vector4f& color ) :
    GLDrawable( GLPrimitiveType::TRIANGLES, calculator() )
{
    updatePositions( box, worldFromBox );
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
