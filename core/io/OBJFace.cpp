#include "libcgt/core/io/OBJFace.h"

OBJFace::OBJFace( bool hasTextureCoordinates, bool hasNormals ) :
    m_bHasTextureCoordinates( hasTextureCoordinates ),
    m_bHasNormals( hasNormals )
{

}

bool OBJFace::hasTextureCoordinates() const
{
    return m_bHasTextureCoordinates;
}

bool OBJFace::hasNormals() const
{
    return m_bHasNormals;
}

int OBJFace::numVertices() const
{
    return static_cast< int >( m_positionIndices.size() );
}

const std::vector< int >& OBJFace::positionIndices() const
{
    return m_positionIndices;
}

std::vector< int >& OBJFace::positionIndices()
{
    return m_positionIndices;
}

std::vector< int >& OBJFace::textureCoordinateIndices()
{
    return m_textureCoordinateIndices;
}

const std::vector< int >& OBJFace::textureCoordinateIndices() const
{
    return m_textureCoordinateIndices;
}

std::vector< int >& OBJFace::normalIndices()
{
    return m_normalIndices;
}

const std::vector< int >& OBJFace::normalIndices() const
{
    return m_normalIndices;
}
