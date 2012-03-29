#include "io/OBJFace.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

OBJFace::OBJFace() :

	m_bHasTextureCoordinates( false ),
	m_bHasNormals( false )

{

}

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

QVector< int >* OBJFace::getPositionIndices()
{
	return &m_qvPositionIndices;
}

QVector< int >* OBJFace::getTextureCoordinateIndices()
{
	return &m_qvTextureCoordinateIndices;
}

QVector< int >* OBJFace::getNormalIndices()
{
	return &m_qvNormalIndices;
}
