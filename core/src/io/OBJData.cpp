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
	foreach( OBJGroup* pGroup, m_qhGroups )
	{
		delete pGroup;
	}
}

QVector< Vector3f >* OBJData::getPositions()
{
	return &m_qvPositions;
}

QVector< Vector2f >* OBJData::getTextureCoordinates()
{
	return &m_qvTextureCoordinates;
}

QVector< Vector3f >* OBJData::getNormals()
{
	return &m_qvNormals;
}

QHash< QString, OBJGroup* >* OBJData::getGroups()
{
	return &m_qhGroups;
}

OBJGroup* OBJData::addGroup( QString groupName )
{
	if( !( m_qhGroups.contains( groupName ) ) )
	{
		m_qhGroups.insert( groupName, new OBJGroup( groupName ) );
	}

	return m_qhGroups[ groupName ];
}

OBJGroup* OBJData::getGroup( QString groupName )
{
	return m_qhGroups[ groupName ];
}

bool OBJData::containsGroup( QString groupName )
{
	return m_qhGroups.contains( groupName );
}
