#pragma once

#include "common/Array1D.h"
#include "common/Array1DView.h"
#include "vecmath/Box3f.h"
#include "vecmath/Box3i.h"
#include "vecmath/Matrix4f.h"
#include "vecmath/Vector2f.h"
#include "vecmath/Vector3f.h"
#include "vecmath/Vector3i.h"
#include "vecmath/Vector4f.h"

namespace libcgt { namespace core { namespace geometry {

// TODO(jiawen): return an error if the buffer isn't big enough?
// Writes a triangle list tesselation of the 6 faces of the box into
// "vertexPositions".
// 6 faces * 2 triangles / face * 3 vertices / triangle = 36 vertices.
// This function is equivalent to combining corners() and
// solidBoxTriangleListIndices().
void writeTriangleListPositions( const Box3f& box,
    Array1DView< Vector4f > positions );

// Writes a triangle list tesselation of the 6 faces of the box, transformed by
// "worldFromBox", into "vertexPositions".
// 6 faces * 2 triangles / face * 3 vertices / triangle = 36 vertices.
// This function is equivalent to combining corners() and
// solidBoxTriangleListIndices().
void writeTriangleListPositions( const Box3f& box, const Matrix4f& worldFromBox,
    Array1DView< Vector4f > positions );

void writeTriangleListNormals( const Box3f& box,
    Array1DView< Vector3f > normals );

void writeTriangleListNormals( const Box3f& box,
    const Matrix4f& worldFromBox, Array1DView< Vector3f > normals );

// Assign somewhat arbitrary texture coordinates to each face of a cube.
// Each face is assigned the same [0,1]^2 domain. The front, right, back, and
// left sides are such that when viewed with x right and y up, the texture has
// u mapped to x and v mapped to y. The top face has u mapped to x and v mapped
// to -z. The bottom face has u mapped to x and v mapped to z.
void writeAxisAlignedSolidTextureCoordinates(
    Array1DView< Vector2f > vertexTextureCoordinates );

// Writes a line list of the 12 edges of the box into "vertexPositions".
// 12 edges * 2 vertices / edge = 24 vertices.
// This function is equivalent to combining corners() and
// wireframeBoxLineListIndices().
void writeWireframe( const Box3f& box,
    Array1DView< Vector4f > vertexPositions );

// Writes a line list of the 12 edges of the box, transformed by
// "worldFromBox", into "vertexPositions".
// 12 edges * 2 vertices / edge = 24 vertices.
// This function is equivalent to combining corners() and
// wireframeBoxLineListIndices().
void writeWireframe( const Box3f& box, const Matrix4f& worldFromBox,
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
void writeWireframeGrid( const Box3f& box, const Vector3i& resolution,
    Array1DView< Vector4f > vertexPositions );

// Writes a line list of a 3D grid subdividing the box divided into
// resolution.xyz bins along each direction. Each coordinate is transformed by
// worldFromBox.
//
// Writes nVertices =
// 2 * (
//         ( resolution.x + 1 ) * ( resolution.y + 1 )
//       + ( resolution.y + 1 ) * ( resolution.z + 1 )
//       + ( resolution.z + 1 ) * ( resolution.x + 1 )
//     )
void writeWireframeGrid( const Box3f& box, const Vector3i& resolution,
    const Matrix4f& worldFromBox, Array1DView< Vector4f > vertexPositions );

// Returns a cube with the given center and side length.
Box3f makeBox( const Vector3f& center, float sideLength );

// Make a box given a center and side lengths.
Box3f makeBox( const Vector3f& center, const Vector3f& sideLengths );

// Returns the 8 corners of the box in hypercube order:
// x changes most frequently, y next, then z least frequently.
Array1D< Vector4f > corners( const Box3f& b );

// Returns the 8 corners of the box in hypercube order:
// x changes most frequently, y next, then z least frequently.
Array1D< Vector3i > corners( const Box3i& b );

// Returns the "diagonal" normals of the 8 corners of the box (pointing
// diagonally out from the center) in hypercube order:
// x changes most frequently, y next, then z least frequently.
Array1D< Vector3f > cornerNormalsDiagonal( const Box3f& b );

// Returns the indices in corners() of a solid box with triangle list topology.
// 6 faces * 2 triangles / face * 3 vertices / triangle = 36 indices.
Array1D< int > solidBoxTriangleListIndices();

// Returns the indices in corners() of a solid box with line list topology.
// 12 edges * 2 vertices / edge = 24 vertices.
Array1D< int > wireframeBoxLineListIndices();

} } } // geometry, core, libcgt
