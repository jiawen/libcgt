#pragma once

#include <vector>

#include "libcgt/core/vecmath/Vector2f.h"
#include "libcgt/core/vecmath/Vector3f.h"
#include "libcgt/core/vecmath/Vector4f.h"

#include "libcgt/core/vecmath/Vector3i.h"

class Icosahedron
{
public:

    // Make an icosahedron centered at center with edge length of 2.
    // All vertices are at a distance sqrt( 1 + phi^2 ) from the origin.
    // To get a "radius" of 1, set scale to 1 / sqrt( 1 + phi^2 ).
    Icosahedron( float scale = 1.0f,
        const Vector3f& center = Vector3f( 0, 0, 0 ) );

    const std::vector< Vector3f >& positions() const;
    const std::vector< Vector3f >& normals() const;
    const std::vector< Vector3i >& faces() const;

    // make a position triangle list of positions and normals (of length 60 each)
    void makeTriangleList( std::vector< Vector4f >& positions,
        std::vector< Vector3f >& normals );

private:

    std::vector< Vector3f > m_positions;
    std::vector< Vector3f > m_normals;
    std::vector< Vector3i > m_faces;

    static Vector3f s_defaultPositions[12];
    static Vector3i s_faces[20];
};
