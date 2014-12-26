#include "geometry/Icosahedron.h"

#include "math/MathUtils.h"

Vector3f Icosahedron::s_defaultPositions[12] =
{
	Vector3f(               0,               1, -MathUtils::PHI ),
	Vector3f(               1,  MathUtils::PHI,               0 ),
	Vector3f(              -1,  MathUtils::PHI,               0 ),
	Vector3f(               0,               1,  MathUtils::PHI ),
	Vector3f(               0,              -1,  MathUtils::PHI ),
	Vector3f( -MathUtils::PHI,               0,               1 ),
	Vector3f(               0,              -1, -MathUtils::PHI ),
	Vector3f(  MathUtils::PHI,               0,              -1 ),
	Vector3f(  MathUtils::PHI,               0,               1 ),
	Vector3f( -MathUtils::PHI,               0,              -1 ),
	Vector3f(               1, -MathUtils::PHI,               0 ),
	Vector3f(              -1, -MathUtils::PHI,               0 )
};

Vector3i Icosahedron::s_faces[20] =
{
	{  0,  1,  2 },
	{  3,  2,  1 },
	{  3,  4,  5 },
	{  3,  8,  4 },
	{  0,  6,  7 },
	{  0,  9,  6 },
	{  4, 10, 11 },
	{  6, 11, 10 },
	{  2,  5,  9 },
	{ 11,  9,  5 },
	{  1,  7,  8 },
	{ 10,  8,  7 },
	{  3,  5,  2 },
	{  3,  1,  8 },
	{  0,  2,  9 },
	{  0,  7,  1 },
	{  6,  9, 11 },
	{  6, 10,  7 },
	{  4, 11,  5 },
	{  4,  8, 10 }
};

Icosahedron::Icosahedron( float scale, const Vector3f& center ) :

	m_positions( 12 ),
	m_normals( 12 ),
	m_faces( s_faces, s_faces + 20 )

{
	for( int i = 0; i < 12; ++i )
	{
		m_positions[i] = center + scale * s_defaultPositions[i];
		m_normals[i] = s_defaultPositions[i].normalized();
	}
}

const std::vector< Vector3f >& Icosahedron::positions() const
{
	return m_positions;
}

const std::vector< Vector3f >& Icosahedron::normals() const
{
	return m_normals;
}

const std::vector< Vector3i >& Icosahedron::faces() const
{
	return m_faces;
}

void Icosahedron::makeTriangleList( std::vector< Vector4f >& positions,
	std::vector< Vector3f >& normals )
{
	positions.resize( 60 );
	normals.resize( 60 );

	for( int i = 0; i < 20; ++i )
	{
		Vector3i face = m_faces[i];
		for( int j = 0; j < 3; ++j )
		{
			positions[ 3 * i + j ] = Vector4f( m_positions[ face[ j ] ], 1 );
			normals[ 3 * i + j ] = m_normals[ face[ j ] ];
		}
	}
}
