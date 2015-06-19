#pragma once

#include <vector>

#include "vecmath/Vector2f.h"
#include "vecmath/Vector3f.h"
#include "vecmath/Vector4f.h"

#include "vecmath/Vector3i.h"

class Icosahedron
{
public:

	// make an icosahedron centered at center
	// with edge length of 2
	// (all vertices are distance sqrt( 1 + phi^2 ) from the origin,
	// set scale to 1 / sqrt( 1 + phi^2 ) to get "radius" 1
	Icosahedron( float scale = 1.0f, const Vector3f& center = Vector3f( 0, 0, 0 ) );

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
