#include "WireframeBox.h"

#include <common/ArrayUtils.h>
#include <geometry/BoxUtils.h>
#include <PlanarVertexBufferCalculator.h>
#include <GLPrimitiveType.h>

using libcgt::core::arrayutils::fill;
using libcgt::core::geometry::boxutils::writeWireframe;

WireframeBox::WireframeBox( const Box3f& box, const Vector4f& color ) :
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
    calculator.addAttribute( 4, sizeof( float ) );
    calculator.addAttribute( 4, sizeof( float ) );
    return calculator;
}
