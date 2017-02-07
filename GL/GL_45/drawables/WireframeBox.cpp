#include "WireframeBox.h"

#include "libcgt/core/common/ArrayUtils.h"
#include "libcgt/core/geometry/BoxUtils.h"
#include "libcgt/GL/PlanarVertexBufferCalculator.h"
#include "libcgt/GL/GLPrimitiveType.h"

using libcgt::core::arrayutils::fill;
using libcgt::core::geometry::writeWireframe;

WireframeBox::WireframeBox( const Box3f& box, const Matrix4f& worldFromBox,
    const Vector4f& color ) :
    GLDrawable( GLPrimitiveType::LINES, calculator() )
{
    updatePositions( box );
    updateColors( color );
}

void WireframeBox::updatePositions( const Box3f& box,
    const Matrix4f& worldFromBox  )
{
    auto mb = mapAttribute< Vector4f >( 0 );
    writeWireframe( box, worldFromBox, mb.view() );
}

void WireframeBox::updateColors( const Vector4f& color )
{
    auto mb = mapAttribute< Vector4f >( 1 );
    fill( mb.view(), color );
}

// static
PlanarVertexBufferCalculator WireframeBox::calculator()
{
    const int NUM_EDGES = 12;
    const int NUM_VERTICES_PER_EDGE = 2;
    const int NUM_VERTICES = NUM_EDGES * NUM_VERTICES_PER_EDGE;
    PlanarVertexBufferCalculator calculator( NUM_VERTICES );
    calculator.addAttribute< Vector4f >();
    calculator.addAttribute< Vector4f >();
    return calculator;
}
