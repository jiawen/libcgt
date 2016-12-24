#include "PointCloud.h"

#include "libcgt/GL/GLPrimitiveType.h"

PointCloud::PointCloud( int nComponents, int nPoints ) :
    GLDrawable( GLPrimitiveType::POINTS, calculator( nComponents, nPoints ) )
{

}

// static
PlanarVertexBufferCalculator PointCloud::calculator( int nComponents,
    int nPoints )
{
    PlanarVertexBufferCalculator calculator( nPoints );
    calculator.addAttribute( nComponents, sizeof( float ) );
    return calculator;
}
