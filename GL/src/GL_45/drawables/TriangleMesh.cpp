#include "TriangleMesh.h"

#include <GLPrimitiveType.h>

TriangleMeshDrawable::TriangleMeshDrawable( const TriangleMesh& mesh ) :
    GLDrawable( GLPrimitiveType::TRIANGLES, calculator( mesh ) )
{
    {
        auto mb = mapAttribute< Vector3f >( 0 );
        Array1DView< Vector3f > positions = mb.view();
        for( int f = 0; f < mesh.numFaces(); ++f )
        {
            Vector3i positionIndices = mesh.faces()[ f ];
            for( int j = 0; j < 3; ++j )
            {
                int pi = positionIndices[ j ];
                positions[ 3 * f + j ] = mesh.positions()[ pi ];
                Vector3f p = mesh.positions()[ pi ];
            }
        }
    }

    {
        auto mb = mapAttribute< Vector3f >( 1 );
        Array1DView< Vector3f > normals = mb.view();
        for( int f = 0; f < mesh.numFaces(); ++f )
        {
            Vector3i normalIndices = mesh.faces()[ f ];
            for( int j = 0; j < 3; ++j )
            {
                int ni = normalIndices[ j ];
                normals[ 3 * f + j ] = mesh.normals()[ ni ];
            }
        }
    }
}

// static
PlanarVertexBufferCalculator TriangleMeshDrawable::calculator(
    const TriangleMesh& mesh )
{
    const int nVertices = 3 * mesh.numFaces();
    PlanarVertexBufferCalculator calculator( nVertices );
    calculator.addAttribute( 3, sizeof( float ) );
    calculator.addAttribute( 3, sizeof( float ) );
    return calculator;
}
