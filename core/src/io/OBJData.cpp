#include "io/OBJData.h"

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

OBJData::OBJData()
{
	
}

// virtual
OBJData::~OBJData()
{
	foreach( OBJMaterial* pMaterial, m_materials )
	{
		delete pMaterial;
	}
	foreach( OBJGroup* pGroup, m_groups )
	{
		delete pGroup;
	}
}

QVector< Vector3f >* OBJData::getPositions()
{
	return &m_positions;
}

QVector< Vector2f >* OBJData::getTextureCoordinates()
{
	return &m_textureCoordinates;
}

QVector< Vector3f >* OBJData::getNormals()
{
	return &m_normals;
}

QHash< QString, OBJGroup* >* OBJData::getGroups()
{
	return &m_groups;
}

OBJGroup* OBJData::addGroup( QString name )
{
	if( !( m_groups.contains( name ) ) )
	{
		m_groups.insert( name, new OBJGroup( name ) );
	}

	return m_groups[ name ];
}

OBJGroup* OBJData::getGroup( QString name )
{
	return m_groups[ name ];
}

bool OBJData::containsGroup( QString name )
{
	return m_groups.contains( name );
}

OBJMaterial* OBJData::addMaterial( QString name )
{
	if( !( m_materials.contains( name ) ) )
	{
		m_materials.insert( name, new OBJMaterial( name ) );
	}

	return m_materials[ name ];
}

OBJMaterial* OBJData::getMaterial( QString name )
{
	return m_materials[ name ];
}

bool OBJData::containsMaterial( QString name )
{
	return m_materials.contains( name );
}
