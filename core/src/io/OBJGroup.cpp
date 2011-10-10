#include "io/OBJGroup.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

OBJGroup::OBJGroup( QString name ) :

	m_qsName( name )

{

}

QString OBJGroup::getName()
{
	return m_qsName;
}

QVector< OBJFace >* OBJGroup::getFaces()
{
	return &m_qvFaces;
}

bool OBJGroup::hasTextureCoordinates()
{
	return m_bHasTextureCoordinates;
}

void OBJGroup::setHasTextureCoordinates( bool b )
{
	m_bHasTextureCoordinates = b;
}

bool OBJGroup::hasNormals()
{
	return m_bHasNormals;
}

void OBJGroup::setHasNormals( bool b )
{
	m_bHasNormals = b;
}
