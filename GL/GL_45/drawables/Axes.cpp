#include "Axes.h"

Axes::Axes( const Matrix4f& worldFromAxes, float axisLength ) :
    GLDrawable( GLPrimitiveType::LINES, calculator() )
{
    {
        auto mb = mapAttribute< Vector4f >( 0 );
        Array1DWriteView< Vector4f > positions = mb.view();
        positions[ 0 ] = worldFromAxes * Vector4f{ 0, 0, 0, 1 };
        positions[ 1 ] = worldFromAxes * Vector4f{ axisLength, 0, 0, 1 };
        positions[ 2 ] = worldFromAxes * Vector4f{ 0, 0, 0, 1 };
        positions[ 3 ] = worldFromAxes * Vector4f{ 0, axisLength, 0, 1 };
        positions[ 4 ] = worldFromAxes * Vector4f{ 0, 0, 0, 1 };
        positions[ 5 ] = worldFromAxes * Vector4f{ 0, 0, axisLength, 1 };
    }

    {
        auto mb = mapAttribute< Vector4f >( 1 );
        Array1DWriteView< Vector4f > colors = mb.view();
        colors[ 0 ] = Vector4f{ 1, 0, 0, 1 };
        colors[ 1 ] = Vector4f{ 1, 0, 0, 1 };
        colors[ 2 ] = Vector4f{ 0, 1, 0, 1 };
        colors[ 3 ] = Vector4f{ 0, 1, 0, 1 };
        colors[ 4 ] = Vector4f{ 0, 0, 1, 1 };
        colors[ 5 ] = Vector4f{ 0, 0, 1, 1 };
    }
}

// static
PlanarVertexBufferCalculator Axes::calculator()
{
    const int NUM_LINES = 3;
    const int NUM_VERTICES = 2 * NUM_LINES;
    PlanarVertexBufferCalculator calculator( NUM_VERTICES );
    calculator.addAttribute( 4, sizeof( float ) );
    calculator.addAttribute( 4, sizeof( float ) );
    return calculator;
}
