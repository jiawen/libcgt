#include "geometry/Icosahedron.h"

#include "math/MathUtils.h"

// TODO(jiawen): once symbols are properly exported, switch to using the
// symbol.
//using libcgt::core::math::PHI;

#define PHI 1.61803398875f

Vector3i Icosahedron::s_faces[20] =
{
    {  0,  8,  4 },
    {  0,  5, 10 },
    {  2,  4,  9 },
    {  2,  11, 5 },
    {  1,  6,  8 },
    {  1, 10,  7 },
    {  3,  9,  6 },
    {  3,  7, 11 },
    {  0, 10,  8 },
    {  1,  8, 10 },
    {  2,  9, 11 },
    {  3,  9, 11 },
    {  4,  2,  0 },
    {  5,  0,  2 },
    {  6,  1,  3 },
    {  7,  3,  1 },
    {  8,  6,  4 },
    {  9,  4,  6 },
    { 10,  5,  7 },
    { 11,  7,  5 }
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
