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

QVector< OBJGroup* >* OBJData::getGroups()
{
	return &m_groups;
}

QHash< QString, OBJGroup* >* OBJData::getGroupsByName()
{
	return &m_groupsByName;
}

OBJGroup* OBJData::addGroup( QString name )
{
	if( !( m_groupsByName.contains( name ) ) )
	{
		m_groupsByName.insert( name, new OBJGroup( name ) );
		m_groups.append( m_groupsByName[ name ] );
	}

	return m_groupsByName[ name ];
}

OBJGroup* OBJData::getGroupByName( QString name )
{
	return m_groupsByName[ name ];
}

bool OBJData::containsGroup( QString name )
{
	return m_groupsByName.contains( name );
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
