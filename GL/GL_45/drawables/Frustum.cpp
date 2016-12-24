#include "Frustum.h"

#include "libcgt/core/common/ArrayUtils.h"
#include "libcgt/core/geometry/RectangleUtils.h"
#include "libcgt/GL/GLPrimitiveType.h"

using libcgt::core::arrayutils::copy;
using libcgt::core::arrayutils::fill;
using libcgt::core::arrayutils::readViewOf;

Frustum::Frustum( const PerspectiveCamera& camera, const Vector4f& color ) :
    GLDrawable( GLPrimitiveType::LINES, calculator() )
{
    updatePositions( camera );
    updateColor( color );
}

void Frustum::updatePositions( const PerspectiveCamera& camera )
{
    auto mb = mapAttribute< Vector4f >( 0 );
    std::vector< Vector4f > frustum_lines = camera.frustumLines();
    copy( readViewOf( frustum_lines ), mb.view() );
}

void Frustum::updateColor( const Vector4f& color )
{
    auto mb = mapAttribute< Vector4f >( 1 );
    fill( mb.view(), color );
}

// static
PlanarVertexBufferCalculator Frustum::calculator()
{
    const int NUM_LINES = 12;
    const int NUM_VERTICES = 2 * NUM_LINES;
    PlanarVertexBufferCalculator calculator( NUM_VERTICES );
    calculator.addAttribute( 4, sizeof( float ) );
    calculator.addAttribute( 4, sizeof( float ) );
    return calculator;
}
