#pragma once

#include "common/Array1DView.h"
#include "vecmath/Box3f.h"
#include "vecmath/Box3i.h"
#include "vecmath/Vector2f.h"
#include "vecmath/Vector3f.h"
#include "vecmath/Vector3i.h"
#include "vecmath/Vector4f.h"

namespace libcgt
{
namespace core
{
namespace geometry
{
namespace boxutils
{
    // Writes a triangle list tesselation of the 6 faces of the box
    // into vertexPositions.
    // 6 faces * 2 triangles / face * 3 vertices / triangle = 36 vertices.
    void writeAxisAlignedSolidBox( const Box3f& box,
        Array1DView< Vector4f > vertexPositions );

    void writeAxisAlignedSolidBoxTextureCoordinates(
        Array1DView< Vector2f > vertexTextureCoordinates );

    // Writes a line list of the 12 edges of the box into vertexPositions.
    // 12 edges * 2 vertices / edge = 24 vertices.
    void writeAxisAlignedWireframeBox( const Box3f& box,
        Array1DView< Vector4f > vertexPositions );

    // Writes a line list of a 3D grid subdividing the box divided into
    // resolution.xyz bins along each direction.
    // 
    // Writes nVertices = 
    // 2 * (
    //         ( resolution.x + 1 ) * ( resolution.y + 1 )
    //       + ( resolution.y + 1 ) * ( resolution.z + 1 )
    //       + ( resolution.z + 1 ) * ( resolution.x + 1 )
    //     )
    void writeAxisAlignedWireframeGrid( const Box3f& box,
        const Vector3i& resolution, Array1DView< Vector4f > vertexPositions );

    // Returns a standard box (cube) with the given center and side length.
    Box3f makeCube( const Vector3f& center, float sideLength );

    // Returns the 8 corners of the box in hypercube order:
    // x changes most frequently, y next, then z least frequently.
    std::vector< Vector3f > corners( const Box3f& b );

    // Returns the 8 corners of the box in hypercube order:
    // x changes most frequently, y next, then z least frequently.
    std::vector< Vector3i > corners( const Box3i& b );
} // boxutils
} // geometry
} // core
} // libcgt